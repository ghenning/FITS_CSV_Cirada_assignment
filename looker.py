import numpy as np
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as patches

if __name__=="__main__":
	desc = "Playing with FITS images and csv files"
	parser = argparse.ArgumentParser(description=desc)
	optional = parser._action_groups.pop()
	required = parser.add_argument_group('required args')
	required.add_argument("--csv_in",type=str,help="CSV to read")
	required.add_argument("--fitsA",type=str,help="Fits tt0")
	required.add_argument("--fitsB",type=str,help="Fits tt1")
	optional.add_argument("--plot",action="store_true",default=False,
			help="plot image")
	optional.add_argument("--plotareas",action="store_true",default=False,
			help="plot areas around sources")
	optional.add_argument("--sourceplots",action="store_true",default=False,
			help="plot a cut-out of each source")
	parser._action_groups.append(optional)
	args = parser.parse_args()

	# read tt0 and grab some header values
	with fits.open(args.fitsA) as F:
		h0 = F[0].header
	ra0 = h0['CRVAL1'] # RA center pixel
	dec0 = h0['CRVAL2'] # DEC center pixel
	f0 = h0['CRVAL3'] # freq
	f1 = h0['CDELT3'] + f0 # freq tt1
	ra0_n = h0['NAXIS1'] # num of RA pixels	
	ra0_change = h0['CDELT1'] # dist between RA pixels
	ra0_ref = h0['CRPIX1'] # ref pixel RA
	ra0_lo = ra0 - int(.5 * ra0_n) * abs(ra0_change) # min RA value
	ra0_hi = ra0 + int(.5 * ra0_n) * abs(ra0_change) # max RA value	
	dec0_n = h0['NAXIS2'] # num of DEC pixels
	dec0_change = h0['CDELT2'] # dist between DEC pixels
	dec0_ref = h0['CRPIX2'] # ref pixel DEC
	dec0_lo = dec0 - int(.5 * dec0_n) * abs(dec0_change) # min DEC value
	dec0_hi = dec0 + int(.5 * dec0_n) * abs(dec0_change) # max DEC value

	# beam info
	bmaj = h0['BMAJ'] # deg
	bmin = h0['BMIN'] #deg
	bang = h0['BPA'] #deg

	# grab data
	dat0 = fits.getdata(args.fitsA,ext=0)
	dat0 = np.squeeze(dat0) # redundant dims
	dat1 = fits.getdata(args.fitsB,ext=0)
	dat1 = np.squeeze(dat1) # redundant dims

	# plot fits
	if args.plot:
		fig,ax = plt.subplots()
		ax.imshow(dat0,aspect='auto',cmap='viridis',origin='lower',norm=mcolors.Normalize(vmin=0,vmax=.00095))

	# read csv and cut to RA DEC range of FITS
	df = pd.read_csv(args.csv_in)
	df_cut = df[(df['RA_Source']>ra0_lo) & (df['RA_Source']<ra0_hi) & (df['DEC_Source']>dec0_lo) & (df['DEC_Source']<dec0_hi)]

	# header of out-csv file
	# RA and DEC of source
	# Peak pixel flux density in image and from catalog in mJy/beam
	# Difference in flux density: S(image) - S(catalog)
	# spectral index (gamma) where applicable
	# is the catalogue source visible in the image (visible)
	out = np.array([["RA(deg)","DEC(deg)","PeakFluxSource(mJy/beam)","PeakFluxSourceCat(mJy/beam)","DeltaS(mJy/beam)","gamma","visible"]])

	# iterate through sources within RA-DEC range of image
	for idx,row in df_cut.iterrows():
		tmp_ra = row['RA_Source'] # RA of source
		tmp_dec = row['DEC_Source'] # DEC of source
		#ang_size_deg = row['Angular_size']/3600 # redundant
		ra_diff = tmp_ra - ra0 # difference between source and central pixel
		ra_diff /= ra0_change # difference in pixels
		ra_diff += ra0_ref # pixel location of source
		ra_diff = int(ra_diff)
		dec_diff = tmp_dec - dec0 # diff between source and central pixel
		dec_diff /= dec0_change # diff in pix
		dec_diff += dec0_ref # pix loc
		dec_diff = int(dec_diff)

		# place an 'x' on each source in the plot
		if args.plot:
			ax.scatter(ra_diff,dec_diff,marker='x',color='m')

		# plot snippets around each source
		if args.sourceplots:
			fig2,ax2 = plt.subplots()
			bbox_ra1 = max(0,ra_diff-50)
			bbox_ra2 = min(ra0_n,ra_diff+50)
			bbox_dec1 = max(0,dec_diff-50)
			bbox_dec2 = min(dec0_n,dec_diff+50)
			cutout = dat0[bbox_dec1:bbox_dec2,bbox_ra1:bbox_ra2]
			ax2.imshow(cutout,aspect='auto',cmap='viridis',origin='lower',norm=mcolors.Normalize(vmin=0,vmax=.00095))
			plt.show()

		# strongest pixel
		# make a little area around the source
		area_ra1 = ra_diff - int(2*bmaj/ra0_change)
		area_ra2 = ra_diff + int(2*bmaj/ra0_change)
		area_dec1 = dec_diff - int(2*bmaj/dec0_change)
		area_dec2 = dec_diff + int(2*bmaj/dec0_change)
		# cut out the area
		cutty = dat0[area_dec1:area_dec2,area_ra2:area_ra1]
		cutty1 = dat1[area_dec1:area_dec2,area_ra2:area_ra1]
		# plot area
		if args.plotareas:
			ax.add_patch(patches.Rectangle(
				(area_ra1,area_dec1),area_ra2-area_ra1,area_dec2-area_dec1,fill=True,color='yellow',alpha=.4,edgecolor=None))
		# find the strongest pixel
		peak_pixel = np.amax(cutty)
		peak_where = np.where(cutty == np.amax(cutty))
		# find the same pixel in tt1
		peak_pixel1 = cutty1[peak_where]

		#background
		# grab an area close to the source
		bg_ra1 = ra_diff - int(6*bmaj/ra0_change)
		bg_ra2 = ra_diff - int(12*bmaj/ra0_change)
		bg_dec1 = dec_diff - int(3*bmaj/dec0_change) 
		bg_dec2 = dec_diff + int(3*bmaj/dec0_change)
		bg = dat0[bg_dec1:bg_dec2,bg_ra1:bg_ra2]
		bg1 = dat1[bg_dec1:bg_dec2,bg_ra1:bg_ra2]
		# plot area
		if args.plotareas:
			ax.add_patch(patches.Rectangle(
				(bg_ra1,bg_dec1),bg_ra2-bg_ra1,bg_dec2-bg_dec1,fill=True,color='yellow',alpha=.4,edgecolor=None))
		# calculate RMS of area in tt0 and tt1
		bg_rms = np.sqrt((1./len(bg.flatten())) * np.sum(bg.flatten()**2))
		bg1_rms = np.sqrt((1./len(bg1.flatten())) * np.sum(bg1.flatten()**2))

		# 'signal-to-noise'
		# let's define 'detectable' as SN > 5
		SN = peak_pixel/bg_rms
		SN1 = (peak_pixel1/bg1_rms)[0]
		# set 'visible' flag to 1 if SN>5 (i.e. visible in tt0 image)
		if SN > 5.:
			visible = 1
		else:
			visible = 0

		# spectral index
		# S = f^gamma
		# log10 the thing, then the slope is just delta(S)/delta(f)
		S0 = peak_pixel - bg_rms
		S1 = peak_pixel1[0] - bg1_rms
		# if 'visible' in tt1 then we can calculate spectral index 
		if SN1 > 5.:
			gamma = (np.log10(S1)-np.log10(S0))/(np.log10(f1)-np.log10(f0))
		else:
			gamma = float("NaN")

		# print some stuff
		#print "bg rms {}".format(bg_rms)
		print "peak pixel flux {:.3f} mJy/b\tcatalog {:.3f} mJy/b".format(1e3*(peak_pixel-bg_rms),row['Peak_flux_source'])
		#print "peak pixel flux tt1 {} mJy/b".format(1e3*(peak_pixel1-bg1_rms))
		print "s/n {}\ts/n1 {}".format(SN,SN1)
		print "gamma {}".format(gamma)
		print "\n"
		#print f0 
		#print h0['CDELT3']
		#print f1 

		# outputs to csv file
		out_plus = np.array([[tmp_ra,tmp_dec,1e3*S0,row['Peak_flux_source'],1e3*S0-row['Peak_flux_source'],gamma,visible]])
		out = np.concatenate((out,out_plus),axis=0)

	# save the csv file
	np.savetxt("test_out.csv",out,delimiter=",",fmt='%s')
	if args.plot:
		plt.show()
	
