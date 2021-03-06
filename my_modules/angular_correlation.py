#! /usr/bin/env python

def make_density_plots(path, zmin, zmax, ngc_data_ra=[], ngc_data_dec =[], sgc_data_ra=[], sgc_data_dec = [],
                       ngc_rand_ra=[], sgc_rand_ra=[], params={}) :
    
    # Import python modules
    import time
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import rc
    
    # Start timing
    start_time = time.time()
    
    # Set up rc parameters
    rc(params)
    
    # Open PDF to store plots
    pp = PdfPages(path)
    
    # Plot NGC data
    plt.figure()
    plt.title(r'NGC Data Galaxy Density, z$\,\in {({:0.3f}, {:0.3f})}$'.format(zmin, zmax))
    plt.xlabel(r'$\alpha\,\cos(\delta)$ [Degrees]')
    plt.ylabel(r'$\delta$ [Degrees]')
    plt.hist2d(ngc_data_ra*np.cos(np.radians(ngc_data_dec)), ngc_data_dec, bins=200)
    plt.colorbar()
    pp.savefig()
    
    # Plot SGC data
    plt.figure()
    plt.title(r'SGC Data Galaxy Density, z$\,\in {({:0.3f}, {:0.3f})}$'.format(zmin, zmax))
    plt.xlabel(r'$\alpha\,\cos(\delta)$ [Degrees]')
    plt.ylabel(r'$\delta$ [Degrees]')
    plt.hist2d(sgc_data_ra*np.cos(np.radians(sgc_data_dec)), sgc_data_dec, bins=200)
    plt.colorbar()
    pp.savefig()
    
    # Plot NGC randoms
    plt.figure()
    plt.title(r'NGC Random Galaxy Density, z$\,\in {({:0.3f}, {:0.3f})}$'.format(zmin, zmax))
    plt.xlabel(r'$\alpha\,\cos(\delta)$ [Degrees]')
    plt.ylabel(r'$\delta$ [Degrees]')
    plt.hist2d(ngc_rand_ra*np.cos(np.radians(ngc_rand_dec)), ngc_rand_dec, bins=200)
    plt.colorbar()
    pp.savefig()
    
    # Plot SGC randoms
    plt.figure()
    plt.title(r'SGC Randoms Galaxy Density, z$\,\in {({:0.3f}, {:0.3f})}$'.format(zmin, zmax))
    plt.xlabel(r'$\alpha\,\cos(\delta)$ [Degrees]')
    plt.ylabel(r'$\delta$ [Degrees]')
    plt.hist2d(sgc_rand_ra*np.cos(np.radians(sgc_rand_dec)), sgc_rand_dec, bins=200)
    plt.colorbar()
    pp.savefig()
    
    # Close the PDF
    pp.close()
    
    # Save the end time
    end_time = time.time()
    
    # Return elapsed time
    return (end_time - start_time)

### This function converts chord distances to great circle distances or great circle angular distances. C is the chord distance, and R is the radius if desired. If R is None (default), the angle is returned in degrees.
def great_circle_dist(C, R=None) :
	import numpy as np
	
	## The formula for the angle is delta(sigma) = 2*arcsin(C/2) according to Wikipedia
	delta_sigma = 2.*np.arcsin(C/2.)
	
	if R is not None :
		return (R*delta_sigma)
	else :
		return (np.rad2deg(delta_sigma))

#### TreeCorr catalog function
def catalog(ra, dec, r=None, perp=False) :
    import treecorr
    import numpy as np

    if perp :
        cat = treecorr.Catalog(ra=ra, dec=dec, r=r, ra_units='deg', dec_units='deg')
    else :
        cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg')
    return (cat)


#### TreeCorr single autocorrelation paircount
def run_TreeCorr_single_auto(ra, dec, config, r=None, perp=False) :
	import treecorr
	import numpy as np

	metric = 'Rperp' if perp else 'Arc'

	# Create the catalogs
	cat = catalog(ra, dec, r=r, perp=perp)

	# Set up NNCorrelation with the config dictionary
	nn = treecorr.NNCorrelation(**config)
	# Run the NNCorrelation process
	# if perp :
	# 	nn.process(cats_corr, metric='Rperp')
	# else :
	# 	nn.process(cats_corr)
	nn.process(cat, metric=metric)
	# Save the results
	# results = {'npairs': nn.npairs,
	# 		   'tot': nn.tot,
	# 		   'logr': nn.logr,
	# 		   'meanlogr': nn.meanlogr,
	# 		   'b': nn.b,
	# 		   'bin_size': nn.bin_size,
	# 		   'bin_slop': nn.bin_slop,
	# 		   'config': nn.config,
	# 		   'corr': nn.corr,
	# 		   'log_sep_units': nn.log_sep_units,
	# 		   'max_sep': nn.max_sep,
	# 		   'min_sep': nn.min_sep,
	# 		   'sep_units': nn.sep_units}
	# # Return the results
	# return (results)
	return (nn.npairs, nn.rnom)


#### TreeCorr single cross correlation paircount
def run_TreeCorr_single_cross(ra1, dec1, ra2, dec2, config, r1=None, r2=None, perp=False) :
	import treecorr
	import numpy as np

	metric = 'Rperp' if perp else 'Arc'

	# Create the catalogs
	cat1 = catalog(ra1, dec1, r=r1, perp=perp)
	cat2 = catalog(ra2, dec2, r=r2, perp=perp)

	# Set up NNCorrelation with the config dictionary
	nn = treecorr.NNCorrelation(**config)
	# Run the NNCorrelation process
	# if perp :
	# 	nn.process(cats_corr1, cats_corr2, metric='Rperp')
	# else :
	# 	nn.process(cats_corr1, cats_corr2)
	nn.process(cat1, cat2, metric=metric)
	# Save the results
	# results = {'npairs': nn.npairs,
	# 		   'tot': nn.tot,
	# 		   'logr': nn.logr,
	# 		   'meanlogr': nn.meanlogr,
	# 		   'b': nn.b,
	# 		   'bin_size': nn.bin_size,
	# 		   'bin_slop': nn.bin_slop,
	# 		   'config': nn.config,
	# 		   'corr': nn.corr,
	# 		   'log_sep_units': nn.log_sep_units,
	# 		   'max_sep': nn.max_sep,
	# 		   'min_sep': nn.min_sep,
	# 		   'sep_units': nn.sep_units}
	# 
	# # Return the results
	# return (results)
	return (0.5*nn.npairs, nn.rnom)


#### Run TreeCorr for all pairings for a set of data and randoms
def paircount_treecorr(data_ra, data_dec, rand_ra, rand_dec, config, file_name, nfile_name, data_r=None, rand_r=None, ret_time=False, min_rpar=None, max_rpar=None) :
	# Import modules
	import time
	import numpy as np
	import sys, os

	if ret_time :
		# Start timing
		start_time = time.time()

	perp = False
	if data_r is not None or rand_r is not None :
		assert data_r is not None and rand_r is not None, 'Must provide distances for both data and randoms or neither'
		perp = True

	assert (min_rpar is None and max_rpar is None) or (min_rpar is not None and max_rpar is not None), 'Must give either both min_rpar and max_rpar or neither'
	if min_rpar is not None :
		config['min_rpar'] = min_rpar
		config['max_rpar'] = max_rpar

	nd = data_ra.size
	nr = rand_ra.size
	dd, r = run_TreeCorr_single_auto(data_ra, data_dec, config, r=data_r, perp=perp)
	rr = run_TreeCorr_single_auto(rand_ra, rand_dec, config, r=rand_r, perp=perp)[0]
	dr = run_TreeCorr_single_cross(data_ra, data_dec, rand_ra, rand_dec, config, r1=data_r, r2=rand_r, perp=perp)[0]
	rd = run_TreeCorr_single_cross(rand_ra, rand_dec, data_ra, data_dec, config, r1=rand_r, r2=data_r, perp=perp)[0]
	
	## Now get the negative r_parallel pair counts, also
	if min_rpar is not None :
		config['min_rpar'] = -max_rpar
		config['max_rpar'] = -min_rpar
		dd += run_TreeCorr_single_auto(data_ra, data_dec, config, r=data_r, perp=perp)[0]
		rr += run_TreeCorr_single_auto(rand_ra, rand_dec, config, r=rand_r, perp=perp)[0]
		dr += run_TreeCorr_single_cross(data_ra, data_dec, rand_ra, rand_dec, config, r1=data_r, r2=rand_r, perp=perp)[0]
		rd += run_TreeCorr_single_cross(rand_ra, rand_dec, data_ra, data_dec, config, r1=rand_r, r2=data_r, perp=perp)[0]
	header = '{:<25s} {:<25s} {:<25s} {:<25s} {:<25s}'.format('# r (Mpc)', 'DD', 'RR', 'DR', 'RD')
	np.savetxt(file_name, np.array([r, dd, rr, dr, rd]).T, fmt='%-25.18f', header=header, comments='')
	np.savetxt(nfile_name, np.array([nd, nr]).T, fmt='%-10d', header='{:<10s} {:<10s}'.format('# N_D', 'N_R'), comments='')

	if ret_time :
		elapsed_time = time.time() - start_time
		del start_time

		# Return results
		return (elapsed_time)
	else :
		return ()


#### Run TreeCorr over KMeans Jackknife regions
def paircount_treecorr_regions(data_ra, data_dec, data_cent, rand_ra, rand_dec, rand_cent, ncen, nbins, config, file_name, file_ext, nfile_name,
                               data_r=None, rand_r=None, ret_time=False, min_rpar=None, max_rpar=None) :
	# Import modules
	import time
	import numpy as np
	import sys, os

	perp = False
	if data_r is not None or rand_r is not None :
		assert data_r is not None and rand_r is not None, 'Must provide distances for both data and randoms or neither'
		perp = True
	# print perp

	assert (min_rpar is None and max_rpar is None) or (min_rpar is not None and max_rpar is not None), 'Must give either both min_rpar and max_rpar or neither'

	ndi, nri = np.empty(ncen, dtype=int), np.empty(ncen, dtype=int)

	if ret_time :
		# Start timing
		start_time = time.time()

	for i in range(ncen) :
		data_i = (data_cent == i)
		rand_i = (rand_cent == i)
		drai = data_ra[data_i].copy()
		ddeci = data_dec[data_i].copy()
		rrai = rand_ra[rand_i].copy()
		rdeci = rand_dec[rand_i].copy()
		if perp :
			dri = data_r[data_i].copy()
			rri = rand_r[rand_i].copy()
		else :
			dri = None
			rri = None
		ndi[i] = drai.size
		nri[i] = rrai.size
		for j in range(i, ncen) :
			data_j = (data_cent == j)
			rand_j = (rand_cent == j)
			draj = data_ra[data_j].copy()
			ddecj = data_dec[data_j].copy()
			rraj = rand_ra[rand_j].copy()
			rdecj = rand_dec[rand_j].copy()
			if perp :
				drj = data_r[data_j].copy()
				rrj = rand_r[rand_j].copy()
			else :
				drj = None
				rrj = None
			## Use r_parallel bin
			if min_rpar is not None :
				config['min_rpar'] = min_rpar
				config['max_rpar'] = max_rpar
			if j == i :
				dd, r = run_TreeCorr_single_auto(drai, ddeci, config, r=dri, perp=perp)
				rr = run_TreeCorr_single_auto(rrai, rdeci, config, r=rri, perp=perp)[0]
			else :
				dd, r = run_TreeCorr_single_cross(drai, ddeci, draj, ddecj, config, r1=dri, r2=drj, perp=perp)
				rr = run_TreeCorr_single_cross(rrai, rdeci, rraj, rdecj, config, r1=rri, r2=rrj, perp=perp)[0]
			dr = run_TreeCorr_single_cross(drai, ddeci, rraj, rdecj, config, r1=dri, r2=rrj, perp=perp)[0]
			rd = run_TreeCorr_single_cross(rrai, rdeci, draj, ddecj, config, r1=rri, r2=drj, perp=perp)[0]
			## Now get negative r_parallel bin
			if min_rpar is not None :
				config['min_rpar'] = -max_rpar
				config['max_rpar'] = -min_rpar
				if j == i :
					dd += run_TreeCorr_single_auto(drai, ddeci, config, r=dri, perp=perp)[0]
					rr += run_TreeCorr_single_auto(rrai, rdeci, config, r=rri, perp=perp)[0]
				else :
					dd += run_TreeCorr_single_cross(drai, ddeci, draj, ddecj, config, r1=dri, r2=drj, perp=perp)[0]
					rr += run_TreeCorr_single_cross(rrai, rdeci, rraj, rdecj, config, r1=rri, r2=rrj, perp=perp)[0]
				dr += run_TreeCorr_single_cross(drai, ddeci, rraj, rdecj, config, r1=dri, r2=rrj, perp=perp)[0]
				rd += run_TreeCorr_single_cross(rrai, rdeci, draj, ddecj, config, r1=rri, r2=drj, perp=perp)[0]
			header = '{:<25s} {:<25s} {:<25s} {:<25s} {:<25s}'.format('# r (Mpc)', 'DD', 'RR', 'DR', 'RD')
			np.savetxt(file_name+'_bin{}_bin{}'.format(i,j)+file_ext, np.array([r, dd, rr, dr, rd]).T, fmt='%-25.18f', header=header, comments='')
			del draj, ddecj, drj, rraj, rdecj, rrj, data_j, rand_j
			del r, dd, rr, dr, rd
		# Delete items
		del data_i, rand_i, drai, ddeci, dri, rrai, rdeci, rri

	np.savetxt(nfile_name, np.array([ndi, nri]).T, fmt='%-10d', header='{:<10s} {:<10s}'.format('# N_D', 'N_R'), comments='')

	if ret_time :
		elapsed_time = time.time() - start_time
		del start_time

		# Return results
		return (elapsed_time)
	else :
		return ()


#### Function to calculate correlation function on full sample
def tc_ang_corr_full(results, ret_time=False) :
    # Import python modules
    import time
    import numpy as np
    import sys
    
    if ret_time :
        # Start timing
        start_time = time.time()
    
    ## Correlation function using full sample
    # DD
    npairs_dd = results['dd']['npairs'].copy()
    tot_dd = results['dd']['tot']
    nd = np.sqrt(2.*tot_dd)
    norm_dd = (nd*(nd - 1.))/2.
    logr = results['dd']['meanlogr'].copy()
    r = np.exp(logr.copy())
    dd = (1.0*npairs_dd.copy())/norm_dd
    ## delete logr
    del logr
    results['dd']['normed'] = dd.copy()
    
    # RR
    npairs_rr = results['rr']['npairs'].copy()
    tot_rr = results['rr']['tot']
    nr = np.sqrt(2.*tot_rr)
    norm_rr = (nr*(nr - 1.))/2.
    rr = (1.0*npairs_rr.copy())/norm_rr
    results['rr']['normed'] = rr.copy()
    
    # DR
    npairs_dr = results['dr']['npairs'].copy()
    norm_dr = (nd*nr)/2.
    dr = (1.0*npairs_dr.copy())/norm_dr
    results['dr']['normed'] = dr.copy()
    
    # RD
    npairs_rd = results['rd']['npairs'].copy()
    norm_rd = (nd*nr)/2.
    rd = (1.0*npairs_rd.copy())/norm_rd
    results['rd']['normed'] = rd.copy()
    
    # Calculate w
    # w = (dd.copy() - 2*dr.copy() + rr.copy())/rr.copy()
    w = (dd.copy()/rr.copy()) - (dr.copy()/rr.copy()) - (rd.copy()/rr.copy()) + 1.0
    results['w'] = w.copy()
    results['r'] = r.copy()
    del npairs_dd, npairs_dr, npairs_rr
    del norm_dd, norm_dr, norm_rr
    
    if ret_time :
        elapsed_time = time.time() - start_time
        del start_time

        # Return results
        return (results, r, w, elapsed_time)
    else :
        return (results, r, w)


#### Calculate correlation function using jackknife regions
def tc_ang_corr_reg(nbins, ncen, results, ret_time=False) :
    # Import python modules
    import time
    import numpy as np
    import sys
    
    if ret_time :
        # Start timing
        start_time = time.time()
    
    ## Correlation function using all regions
    ### Find first non-empty logr to read r from
    class Found(Exception) : pass
    try :
    	for i in range(ncen) :
    		for j in range(i, ncen) :
    			if 'logr' in results['dd'][i][j] :
    				raise Found
    except Found :
    	r = np.exp(results['dd'][i][j]['logr'].copy())
    results['r'] = r.copy()
    npairs_dd = np.zeros(nbins)
    tot_dd = 0
    npairs_dr = np.zeros(nbins)
    tot_dr = 0
    npairs_rd = np.zeros(nbins)
    tot_rd = 0
    npairs_rr = np.zeros(nbins)
    tot_rr = 0
    results['dd']['summed'] = {}
    results['dd']['summed']['full'] = {}
    results['dr']['summed'] = {}
    results['dr']['summed']['full'] = {}
    results['rd']['summed'] = {}
    results['rd']['summed']['full'] = {}
    results['rr']['summed'] = {}
    results['rr']['summed']['full'] = {}
    results['w'] = {}
    for i in range(ncen) :
        for j in range(i, ncen) :
            npairs_dd += results['dd'][i][j]['npairs'].copy()
            tot_dd += results['dd'][i][j]['tot']
            npairs_dr += results['dr'][i][j]['npairs'].copy()
            tot_dr += results['dr'][i][j]['tot']
            npairs_rd += results['rd'][i][j]['npairs'].copy()
            tot_rd += results['rd'][i][j]['tot']
            npairs_rr += results['rr'][i][j]['npairs'].copy()
            tot_rr += results['rr'][i][j]['tot']
    nd = np.sqrt(2.*tot_dd)
    nr = np.sqrt(2.*tot_rr)
    norm_dd = (nd*(nd - 1.))/2.
    norm_dr = (nd*nr)/2.
    norm_rd = (nd*nr)/2.
    norm_rr = (nr*(nr - 1.))/2.
    dd = (1.0*npairs_dd.copy())/norm_dd
    dr = (1.0*npairs_dr.copy())/norm_dr
    rd = (1.0*npairs_rd.copy())/norm_rd
    rr = (1.0*npairs_rr.copy())/norm_rr
    results['dd']['summed']['full']['npairs'] = 1.0*npairs_dd.copy()
    results['dd']['summed']['full']['tot'] = nd
    results['dd']['summed']['full']['norm'] = norm_dd
    results['dr']['summed']['full']['npairs'] = 1.0*npairs_dr.copy()
    results['dr']['summed']['full']['norm'] = norm_dr
    results['rd']['summed']['full']['npairs'] = 1.0*npairs_rd.copy()
    results['rd']['summed']['full']['norm'] = norm_rd
    results['rr']['summed']['full']['npairs'] = 1.0*npairs_rr.copy()
    results['rr']['summed']['full']['tot'] = nr
    results['rr']['summed']['full']['norm'] = norm_rr
    w_reg = (dd.copy()/rr.copy()) - (dr.copy()/rr.copy()) - (rd.copy()/rr.copy()) + 1.0
    results['w']['all'] = w_reg.copy()
    del npairs_dd, npairs_dr, npairs_rd, npairs_rr
    del tot_dd, tot_dr, tot_rd, tot_rr
    del norm_dd, norm_dr, norm_rd, norm_rr
    del nd, nr
    del dd, dr, rd, rr
    
    ## Now find the errors
    w_i = np.empty((ncen, nbins))
    for i in range(ncen) :
    	results['dd']['summed'][i] = {}
    	results['dr']['summed'][i] = {}
    	results['rd']['summed'][i] = {}
    	results['rr']['summed'][i] = {}
        npairs_dd = np.zeros_like(w_reg)
        tot_dd = 0
        npairs_dr = np.zeros_like(w_reg)
        tot_dr = 0
        npairs_rd = np.zeros_like(w_reg)
        tot_rd = 0
        npairs_rr = np.zeros_like(w_reg)
        tot_rr = 0
        for j in range(ncen) :
            if not j == i :
                for k in range(j, ncen) :
                    if not k == i :
                        npairs_dd += results['dd'][j][k]['npairs'].copy()
                        tot_dd += results['dd'][j][k]['tot']
                        npairs_dr += results['dr'][j][k]['npairs'].copy()
                        tot_dr += results['dr'][j][k]['tot']
                        npairs_rd += results['rd'][j][k]['npairs'].copy()
                        tot_rd += results['rd'][j][k]['tot']
                        npairs_rr += results['rr'][j][k]['npairs'].copy()
                        tot_rr += results['rr'][j][k]['tot']
        nd = np.sqrt(2.*tot_dd)
        nr = np.sqrt(2.*tot_rr)
        norm_dd = (nd*(nd - 1.))/2.
        norm_dr = (nd*nr)/2.
        norm_rd = (nd*nr)/2.
        norm_rr = (nr*(nr - 1.))/2.
        dd = (1.0*npairs_dd.copy())/norm_dd
        dr = (1.0*npairs_dr.copy())/norm_dr
        rd = (1.0*npairs_rd.copy())/norm_rd
        rr = (1.0*npairs_rr.copy())/norm_rr
        results['dd']['summed'][i]['npairs'] = 1.0*npairs_dd.copy()
        results['dd']['summed'][i]['tot'] = nd
        results['dd']['summed'][i]['norm'] = norm_dd
        results['dr']['summed'][i]['npairs'] = 1.0*npairs_dr.copy()
        results['dr']['summed'][i]['norm'] = norm_dr
        results['rd']['summed'][i]['npairs'] = 1.0*npairs_rd.copy()
        results['rd']['summed'][i]['norm'] = norm_rd
        results['rr']['summed'][i]['npairs'] = 1.0*npairs_rr.copy()
        results['rr']['summed'][i]['tot'] = nr
        results['rr']['summed'][i]['norm'] = norm_rr
        ## delete 
        del npairs_dd, npairs_dr, npairs_rd, npairs_rr
        del tot_dd, tot_dr, tot_rd, tot_rr
        w_i[i] = (dd.copy()/rr.copy()) - (dr.copy()/rr.copy()) - (rd.copy()/rr.copy()) + 1.0
        results['w'][i] = w_i[i].copy()
        del dd, dr, rd, rr
        del norm_dd, norm_dr, norm_rd, norm_rr
        del nd, nr
    w_ave = (1./float(ncen))*np.sum(w_i, axis=0)
    sig_sq = (float(ncen-1)/float(ncen))*np.dot((w_i.copy() - w_ave.copy()).T, (w_i.copy() - w_ave.copy()))
    results['cov'] = sig_sq.copy()
    
    if ret_time :
        elapsed_time = time.time() - start_time
        del start_time

        # Return results
        return (results, r, w_reg, sig_sq, elapsed_time)
    else :
        return (results, r, w_reg, sig_sq)

def read_proj_pc(nbins, ncen, file_name, file_ext) :
	import numpy as np
	if ncen == 1 :
		r, dd, rr, dr, rd = np.loadtxt(file_name+file_ext, unpack=True)
	else :
		r = np.loadtxt('{}_bin0_bin0{}'.format(file_name, file_ext), usecols=(0,))
		dd = np.empty((int((ncen*(ncen+1))/2.), r.size), dtype=r.dtype)
		rr, dr, rd = np.empty_like(dd), np.empty_like(dd), np.empty_like(dd)
		idx = 0
		for i in range(ncen) :
			for j in range(i, ncen) :
				dd[idx], rr[idx], dr[idx], rd[idx] = np.loadtxt('{}_bin{}_bin{}'\
				'{}'.format(file_name, i, j, file_ext), usecols=range(1,5), unpack=True)
				idx += 1
	return r, dd, rr, dr, rd

def tc_proj_corr(nbins, file_name, nd, nr) :
	import numpy as np
	import os
	
	## Read from files
	# r, dd, rr, dr, rd = np.loadtxt(file_name, unpack=True)
	r, dd, rr, dr, rd = read_proj_pc(nbins, 0, os.path.splitext(file_name)[0], os.path.splitext(file_name)[1])
	## Find normalizations
	norm_dd = 2./(nd*(nd - 1.))
	norm_rr = 2./(nr*(nr - 1.))
	norm_dr = 2./(nd*nr)
	## Find and return correlation function
	wp = ((norm_dd*dd.copy()) - (norm_dr*dr.copy()) - (norm_dr*rd.copy()) + (norm_rr*rr.copy()))/(norm_rr*rr.copy())
	return (r, wp)

def tc_proj_corr_reg(nbins, ncen, file_name, file_ext, nd, nr) :
	import numpy as np
	import os, sys
	
	## Read from files
	# r = np.loadtxt('{}_bin0_bin0{}'.format(file_name, file_ext), usecols=(0,))
	# ddij = np.empty((int((ncen*(ncen+1))/2.), r.size), dtype=r.dtype)
	# rrij, drij, rdij = np.empty_like(ddij), np.empty_like(ddij), np.empty_like(ddij)
	# idx = 0
	# for i in range(ncen) :
	# 	for j in range(i, ncen) :
	# 		ddij[idx], rrij[idx], drij[idx], rdij[idx] = np.loadtxt('{}_bin{}_bin{}{}'.format(file_name, i, j, file_ext), usecols=range(1,5), unpack=True)
	# 		idx += 1
	r, ddij, rrij, drij, rdij = read_proj_pc(nbins, ncen, file_name, file_ext)
	
	## Compute full correlation function
	norm_dd = 2./(nd.sum()*(nd.sum() - 1.))
	norm_rr = 2./(nr.sum()*(nr.sum() - 1.))
	norm_dr = 2./(nd.sum()*nr.sum())
	### Debugging: check norm_rr for zero, and compare to norm_dd
	print norm_dd
	print norm_rr
	sys.stdout.flush()
	
	dd = norm_dd*ddij.sum(axis=0)
	rr = norm_rr*rrij.sum(axis=0)
	dr = norm_dr*drij.sum(axis=0)
	rd = norm_dr*rdij.sum(axis=0)
	### Debugging: check RR for zeros
	print 'Any DD = 0? {}'.format(np.any(dd == 0))
	print 'Number of zeros in DD = {} out of {}'.format(dd.size - np.count_nonzero(dd), dd.size)
	print 'Any RR = 0? {}'.format(np.any(rr == 0))
	print 'Number of zeros in RR = {} out of {}'.format(rr.size - np.count_nonzero(rr), rr.size)
	sys.stdout.flush()
	wp = (dd.copy() - dr.copy() - rd.copy() + rr.copy())/rr.copy()
	del dd, rr, dr, rd, norm_dd, norm_rr, norm_dr
	
	## Compute jackknife errors
	wi = np.empty((ncen, nbins), dtype=wp.dtype)
	for i in range(ncen) :
		ndi = nd.sum() - nd[i]
		nri = nr.sum() - nr[i]
		norm_dd = 2./(ndi*(ndi - 1.))
		norm_rr = 2./(nri*(nri - 1.))
		norm_dr = 2./(ndi*nri)
		### Debugging: check norm_rr for zero, and compare to norm_dd
		print 'DD_{} normalization = {}'.format(i, norm_dd)
		print 'RR_{} normalization = {}'.format(i, norm_rr)
		sys.stdout.flush()
		ddi, rri, dri, rdi = np.zeros_like(wp), np.zeros_like(wp), np.zeros_like(wp), np.zeros_like(wp)
		idx = 0
		for j in range(ncen) :
			for k in range(j, ncen) :
				if j != i and k != i :
					ddi += ddij[idx].copy()
					rri += rrij[idx].copy()
					dri += drij[idx].copy()
					rdi += rdij[idx].copy()
				idx += 1
		### Debugging: check RR_i for zeros
		print 'Any DD_{} = 0? {}'.format(i, np.any(ddi == 0))
		print 'Number of zeros in DD_{} = {} out of {}'.format(i, ddi.size - np.count_nonzero(ddi), ddi.size)
		print 'Any RR_{} = 0? {}'.format(i, np.any(rri == 0))
		print 'Number of zeros in RR_{} = {} out of {}'.format(i, rri.size - np.count_nonzero(rri), rri.size)
		ddi *= norm_dd
		rri *= norm_rr
		dri *= norm_dr
		rdi *= norm_dr
		wi[i] = (ddi.copy() - dri.copy() - rdi.copy() + rri.copy())/rri.copy()
		del ddi, rri, dri, rdi, ndi, nri, norm_dd, norm_rr, norm_dr
	wi_ave = (1./float(ncen))*wi.sum(axis=0)
	cov = ((ncen - 1.)/float(ncen))*np.dot((wi.copy() - wi_ave.copy()).T, (wi.copy() - wi_ave.copy()))
	err = np.sqrt(np.diag(cov))
	del wi_ave, wi
	
	return (r, wp, err, cov)
	
### Unit test: check (normalized) pair counts against analytic (normalized) counts
### Notes: assumes normalized pair counts are given if normed=True, and compares 
### unnormalized pair counts otherwise
### Use xi = numpy.zeros_like(dd) for random-random pair counts
### Assumes xi is a function of r only, not rperp, rparallel!
def check_counts_1D(rmin, rmax, nbins, Nd, Nr, xi, dd, rr, dr, rd=None, normed=False, runits='Mpc', ret_plots=True) :
	import numpy as np
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib import rc
	from matplotlib.ticker import MaxNLocator
	import sys
	rc('text', usetex=True)
	font_dict = {'family':'serif', 'serif':'cm', 'size':16}
	rc('font', **font_dict)
	
	## bin size, separations
	bs = np.log(rmax/rmin)/float(nbins)
	r = rmin*np.exp((np.arange(nbins)+0.5)*bs)
	## Some stuff for speed
	### Normalizations if we want to compare normalized pair counts. Otherwise, make them all 1
	if normed :
		norm_dd = 2./(float(Nd)*(float(Nd) - 1.))
		norm_rr = 2./(float(Nr)*(float(Nr) - 1.))
		norm_dr = 2./(float(Nd)*float(Nr))
	else :
		norm_dd, norm_rr, norm_dr = 1., 1., 1.
	
	## Vshell
	Vshell = 2.*np.pi*np.power(r, 2)*np.sinh(bs)
	### Survey volume
	volume = np.average(norm_rr*np.power(Nr, 2)*Vshell.copy()/(2.*rr.copy()))
	### number densities
	nd = float(Nd)/volume
	nr = float(Nr)/volume
	
	## If we are returning the pair counts rather than the plots, make sure they are normalized
	if not ret_plots :
		norm_dd = 2./(float(Nd)*(float(Nd) - 1.))
		norm_rr = 2./(float(Nr)*(float(Nr) - 1.))
		norm_dr = 2./(float(Nd)*float(Nr))
	
	## Analytic pair counts
	dd_exp = norm_dd*0.5*np.power(nd,2)*volume*Vshell.copy()*(1.+xi.copy())
	rr_exp = norm_rr*0.5*np.power(nr,2)*volume*Vshell.copy()
	dr_exp = norm_dr*0.5*nd*nr*volume*Vshell.copy()
	
	if ret_plots :
		## 1/(expected pair counts): for residuals
		one_over_dd = 1./dd_exp.copy()
		one_over_rr = 1./rr_exp.copy()
		one_over_dr = 1./dr_exp.copy()
	
		## Plot pair counts
		### DD
		fig_dd = plt.figure(figsize=(10,6), facecolor='w')
		frame_dd1 = fig_dd.add_axes((.1,.4,.8,.5))
		frame_dd1.set_xscale('log')
		frame_dd1.set_xlim([rmin, rmax])
		plt.title(r'DD')
		frame_dd1.set_ylabel(r'$DD(r)$')
		plt.plot(r, dd, 'ro', label=r'Observed')
		plt.plot(r, dd_exp, 'b-', lw=1.5, label=r'Analytic')
		plt.legend(loc=0, fontsize=font_dict['size'], fancybox=True, framealpha=0.5)
		nx = len(frame_dd1.get_xticklabels())
		frame_dd1.set_xticklabels([])
		frame_dd2 = fig_dd.add_axes((.1,.1,.8,.3))
		frame_dd2.set_xscale('log')
		frame_dd2.set_xlim([rmin, rmax])
		plt.plot(r, one_over_dd*(dd - dd_exp), 'ro')
		frame_dd2.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='upper'))
		frame_dd1.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='lower'))
		frame_dd2.set_ylabel(r'$DD_o/DD_e - 1$')
		frame_dd2.set_xlabel(r'$r$ {}'.format(runits))
		### RR
		fig_rr = plt.figure(figsize=(10,6), facecolor='w')
		frame_rr1 = fig_rr.add_axes((.1,.4,.8,.5))
		frame_rr1.set_xscale('log')
		frame_rr1.set_xlim([rmin, rmax])
		plt.title(r'RR')
		frame_rr1.set_ylabel(r'$RR(r)$')
		plt.plot(r, rr, 'ro', label=r'Observed')
		plt.plot(r, rr_exp, 'b-', lw=1.5, label=r'Analytic')
		plt.legend(loc=0, fontsize=font_dict['size'], fancybox=True, framealpha=0.5)
		nx = len(frame_rr1.get_xticklabels())
		frame_rr1.set_xticklabels([])
		frame_rr2 = fig_rr.add_axes((.1,.1,.8,.3))
		frame_rr2.set_xscale('log')
		frame_rr2.set_xlim([rmin, rmax])
		plt.plot(r, one_over_rr*(rr - rr_exp), 'ro')
		frame_rr2.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='upper'))
		frame_rr1.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='lower'))
		frame_rr2.set_ylabel(r'$RR_o/RR_e - 1$')
		frame_rr2.set_xlabel(r'$r$ {}'.format(runits))
		### DR
		fig_dr = plt.figure(figsize=(10,6), facecolor='w')
		frame_dr1 = fig_dr.add_axes((.1,.4,.8,.5))
		frame_dr1.set_xscale('log')
		frame_dr1.set_xlim([rmin, rmax])
		plt.title(r'DR')
		frame_dr1.set_ylabel(r'$DR(r)$')
		plt.plot(r, dr, 'ro', label=r'Observed')
		plt.plot(r, dr_exp, 'b-', lw=1.5, label=r'Analytic')
		plt.legend(loc=0, fontsize=font_dict['size'], fancybox=True, framealpha=0.5)
		nx = len(frame_dr1.get_xticklabels())
		frame_dr1.set_xticklabels([])
		frame_dr2 = fig_dr.add_axes((.1,.1,.8,.3))
		frame_dr2.set_xscale('log')
		frame_dr2.set_xlim([rmin, rmax])
		plt.plot(r, one_over_dr*(dr - dr_exp), 'ro')
		frame_dr2.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='upper'))
		frame_dr1.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='lower'))
		frame_dr2.set_ylabel(r'$DR_o/DR_e - 1$')
		frame_dr2.set_xlabel(r'$r$ {}'.format(runits))
		### RD: if any!
		if rd is not None :
			fig_rd = plt.figure(figsize=(10,6), facecolor='w')
			frame_rd1 = fig_rd.add_axes((.1,.4,.8,.5))
			frame_rd1.set_xscale('log')
			frame_rd1.set_xlim([rmin, rmax])
			plt.title(r'RD')
			frame_rd1.set_ylabel(r'$RD(r)$')
			plt.plot(r, rd, 'ro', label=r'Observed')
			plt.plot(r, dr_exp, 'b-', lw=1.5, label=r'Analytic')
			plt.legend(loc=0, fontsize=font_dict['size'], fancybox=True, framealpha=0.5)
			nx = len(frame_rd1.get_xticklabels())
			frame_rd1.set_xticklabels([])
			frame_rd2 = fig_rd.add_axes((.1,.1,.8,.3))
			frame_rd2.set_xscale('log')
			frame_rd2.set_xlim([rmin, rmax])
			plt.plot(r, one_over_dr*(rd - dr_exp), 'ro')
			frame_rd2.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='upper'))
			frame_rd1.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='lower'))
			frame_rd2.set_ylabel(r'$RD_o/RD_e - 1$')
			frame_rd2.set_xlabel(r'$r$ {}'.format(runits))
			return fig_dd, fig_rr, fig_dr, fig_rd
		else :
			return fig_dd, fig_rr, fig_dr
	else :
		return dd_exp, rr_exp, dr_exp

## Part of unit test above: find volume given solid angle in sky and redshifts+errors
### Need z, zerr, cosmology, solid angle, min and max redshift for integration
def volume(sa, z, ze, cosmology, zmin, zmax) :
	from astropy.modeling.models import Gaussian1D
	from ConfigParser import ConfigParser
	from astropy.cosmology import w0waCDM
	from scipy.integrate import quad
	from scipy.interpolate import UnivariateSpline
	import numpy as np
	## Redshift sample for spline
	table_z, dz = np.linspace(0., 2., num=101, retstep=True)
	## Get cosmology
	config = ConfigParser()
	config.read(cosmology)
	cosmo_dict = config._sections['cosmological_parameters']
	for key, value in cosmo_dict.iteritems() :
		if key != '__name__' :
			cosmo_dict[key] = float(value)
	omega_L = 1. - cosmo_dict['omega_m'] - cosmo_dict['omega_k']
	cosmo = w0waCDM(100*cosmo_dict['h0'], cosmo_dict['omega_m'], omega_L, \
	w0=cosmo_dict['w'], wa=cosmo_dict['wa'])
	table_dVc = cosmo.differential_comoving_volume(table_z)
	## Get redshift phi(z)
	amp = 1./np.sqrt(2.*np.pi*ze**2)
	g = Gaussian1D(amplitude=amp, mean=z, stddev=ze)
	table_g = np.array([np.sum(g(zi)) for zi in table_z])
	table_g /= np.sum(table_g*dz)
	## Spline of phi(z)*dVc(z)
	integrand = UnivariateSpline(table_z, table_g*table_dVc, s=0, k=1)
	V = sa*quad(integrand, zmin, zmax)[0]
	return V

### Unit test: check correlation function we would expect analytically with what we 
### actually get. Assumes xi is only a function of one separation, not components
### Notes: uses the check_counts_1D function to get analytic normalized pair counts
def check_xi_1D(xi_found, err, xi_exp, rmin, rmax, nbins, Nd, Nr, dd, rr, dr, normed=False, runits='Mpc') :
	import numpy as np
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib import rc
	from matplotlib.ticker import MaxNLocator
	import sys
	rc('text', usetex=True)
	font_dict = {'family':'serif', 'serif':'cm', 'size':16}
	rc('font', **font_dict)
	
	## Find r bins
	b = np.log(rmax/rmin)/float(nbins)
	r = rmin*np.exp((np.arange(nbins)+0.5)*b)
	
	## Get normalized pair counts
	dd_exp, rr_exp, dr_exp = check_counts_1D(rmin, rmax, nbins, Nd, Nr, xi_exp, dd, \
	rr, dr, rd=None, normed=normed, runits=runits, ret_plots=False)
	
	## Find correlation function
	### For speed:
	one_over_rr = 1./rr_exp.copy()
	xi_calc = one_over_rr.copy()*dd_exp.copy() - 2.*one_over_rr.copy()*dr_exp.copy() + 1.
	
	### For speed:
	one_over_xi = 1./xi_calc.copy()
	# one_over_xi = 1./xi_found.copy()
	# one_over_xisqr = np.power(one_over_xi, 2)
	
	fig = plt.figure(figsize=(10,6), facecolor='w')
	frame1 = fig.add_axes((.1,.4,.8,.5))
	frame1.set_xscale('log')
	frame1.set_xlim([rmin, rmax])
	frame1.set_ylabel(r'$r_\perp^2 w_p(r_\perp)$')
	nx = len(frame1.get_xticklabels())
	frame1.set_xticklabels([])
	plt.errorbar(r, r**2*xi_found, yerr=r**2*err, fmt='ro', label=r'Observed')
	plt.plot(r, r**2*xi_calc, 'b-', lw=1.5, label=r'Analytic')
	plt.legend(loc=0, fontsize=font_dict['size'], fancybox=True, framealpha=0.5)
	frame2 = fig.add_axes((.1,.1,.8,.3))
	frame2.set_xscale('log')
	frame2.set_xlim([rmin, rmax])
	frame2.set_xlabel(r'$r_\perp$ ({})'.format(runits))
	frame2.set_ylabel(r'$w_{p,o}/w_{p,e} - 1$')
	# frame2.set_ylabel(r'$1 - w_{p,e}/w_{p,o}$')
	plt.errorbar(r, one_over_xi*(xi_found - xi_calc), yerr=one_over_xi*xi_found*err, fmt='ro')
	# plt.errorbar(r, one_over_xi*(xi_found - xi_calc), yerr=one_over_xisqr*xi_calc*err, fmt='ro')
	frame1.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='lower'))
	frame2.yaxis.set_major_locator(MaxNLocator(nbins=nx, prune='upper'))
	return fig

#### TreeCorr Xi calculation
def correlation_TreeCorr(data_ra, data_dec, rand_ra, rand_dec, config) :
	import time
	import numpy as np
	import treecorr
	import sys

	# Begin timing
	start = time.time()

	# Make sure arrays match
	assert data_ra.size == data_dec.size, "Data must have both RA and DEC"
	assert rand_ra.size == rand_dec.size, "Randoms must have both RA and DEC"

	# Create TreeCorr catalog objects
	dcat = treecorr.Catalog(ra=data_ra, dec=data_dec, ra_units='deg', dec_units='deg')
	rcat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='deg', dec_units='deg')
	print ('TreeCorr catalogs created')
	sys.stdout.flush()

	# Run TreeCorr processes for DD, DR, RD, and RR
	dd = treecorr.NNCorrelation(config)
	dr = treecorr.NNCorrelation(config)
	# rd = treecorr.NNCorrelation(config)
	rr = treecorr.NNCorrelation(config)
	dd.process(dcat)
	print ('DD done')
	sys.stdout.flush()
	# I also need to get the bin locations for plotting
	logr = dd.logr
	dr.process(dcat, rcat)
	print ('DR done')
	sys.stdout.flush()
	# rd.process(rcat, dcat)
	# print ('RD done')
	# sys.stdout.flush()
	rr.process(rcat)
	print ('RR done')
	sys.stdout.flush()

	# Find the correlation function and errors
	# xi, varxi = dd.calculateXi(rr, dr, rd)
	xi, varxi = dd.calculateXi(rr, dr)
	print ('Correlation function and errors calculated')
	sys.stdout.flush()

	# Find elapsed time
	runtime = time.time() - start
	del start
	## Print the time it took
	h = int(np.floor(runtime/(60.0*60.0)))
	m = int(np.floor((runtime - (60.0*60.0*h))/60.0))
	s = runtime - 60.0*60.0*h - 60.0*m
	print ('Elapsed time: {:>02d}:{:>02d}:{:>05.2f}'.format(h, m, s))
	sys.stdout.flush()
	del runtime, h, m, s

	# Return xi, varxi, and bin locations
	return (xi, varxi, logr)


#### Create parameter file for KDTPCF
def init_kdtpcf_params(param_path, s_min, s_max, nbins, log_bins, data_path, rand_path, out_base, jk_reg=4, nthr=8) :
    # Import python modules
    import time
    import numpy as np
    
    # Start timing
    start_time = time.time()
    
    # Create array of objects that are in the parameter file
    items = np.array(["s_max", "s_min", "s_bin_num", "phi_bin_num", "regular_phi_bin", "log_bin", "file_data", 
                      "file_rand", "out_name_base", "lambda", "z_max", "corr_stat", "weighted_bin", "jackknife_depth",
                      "bin_count_type", "num_threads"])
    inputs = np.array(['{:<0.4f}'.format(s_max), '{:<0.4f}'.format(s_min), '{:<d}'.format(nbins), '{:<d}'.format(40), 
                       '{:<d}'.format(0), '{:<d}'.format(log_bins), data_path, rand_path, out_base, 
                       '{:<0.4f}'.format(0.7), '{:<0.4f}'.format(5), '{:<d}'.format(0), '{:<d}'.format(0), 
                       '{:<d}'.format(jk_reg), '{:<d}'.format(0), '{:<d}'.format(nthr)])
    outputs = np.array([items.copy(), inputs.copy()]).transpose()
    
    hdr = 'CORR_FUNC\n\nInit'
    
    # Output to file
    np.savetxt(param_path, outputs, fmt='%-17s', header=hdr, comments='')
    
    # Stop time
    end_time = time.time()
    
    # Return objects
    return (end_time - start_time)


def read_output(out_base, jk_reg=4) :
    # Import modules
    import time
    import numpy as np
    import sys
    
    # Start timing
    start_time = time.time()
    
    # Initialize dictionary to store results in
    results = {}
    ncen = 2**jk_reg
    
    # Read in data from files
    dd = out_base + '_ddbins'
    dr = out_base + '_drbins'
    rd = out_base + '_rdbins'
    rr = out_base + '_rrbins'
    dd_data = np.loadtxt(dd, dtype=None)
    ## Debugging: currently verifying number of pairs, so only getting out dd
    dr_data = np.loadtxt(dr, dtype=None)
    rd_data = np.loadtxt(rd, dtype=None)
    rr_data = np.loadtxt(rr, dtype=None)
    
    # Store results in the dictionary. I'll only save distances from the dd paircount, and I'll use the central value.
    results['r'] = dd_data[:,1].copy()  # central value of separation for each bin

    results['dd'] = {}
    ## Debugging: currently verifying number of pairs, so only getting out dd
    results['dr'] = {}
    results['rd'] = {}
    results['rr'] = {}
    # npairs will be the normalized paircount in each bin. The jackknife paircounts will be labeled by excluded region
    results['dd']['npairs'] = dd_data[:,3].copy()
    ## Debugging: currently verifying number of pairs, so only getting out dd
    results['dr']['npairs'] = dr_data[:,3].copy()
    results['rd']['npairs'] = rd_data[:,3].copy()
    results['rr']['npairs'] = rr_data[:,3].copy()
    for i in range(ncen) :
        results['dd'][i] = {}
        results['dr'][i] = {}
        results['rd'][i] = {}
        results['rr'][i] = {}
        results['dd'][i]['npairs'] = dd_data[:,i+4].copy()
        results['dr'][i]['npairs'] = dr_data[:,i+4].copy()
        results['rd'][i]['npairs'] = rd_data[:,i+4].copy()
        results['rr'][i]['npairs'] = rr_data[:,i+4].copy()
    
    # Now that I should have all data that I need copied over, I can delete the data arrays
    ## Debugging: currently verifying number of pairs, so only getting out dd
    # del dd_data
    del dd_data, dr_data, rd_data, rr_data
    
    # Stop timing
    end_time = time.time()
    
    # Return objects
    return (results, end_time - start_time)


def paircount_kdtpcf(data_ra, data_dec, rand_ra, rand_dec, s_min, s_max, nbins, log_bins, jk_reg=4, nthr=8, param_path=None) :
    # Import python modules
    import time
    import os
    import sys
    import numpy as np
    import subprocess
    import threading
    import shlex
    
    # Start timing
    start_time = time.time()
    
    # Set the default paths
    default_params = '/home/wagoner47/autocorrelation_code/KDTPCF/params.txt'
    temp_path = '/calvin1/wagoner47/kdtpcf_temp'
    try :
        os.makedirs(temp_path)
    except OSError :
        if not os.path.isdir(temp_path) :
            raise
        else :
            pass
    print ('Temporary directory created')
    sys.stdout.flush()
    default_data = temp_path + '/data'
    default_rand = temp_path + '/rand'
    default_out = temp_path + '/out'
    
    # Check for values in arguments
    if param_path is None :
        param_path = default_params
    
    # Create the data and random files first to make sure they exist
    np.savetxt(default_data, np.array([data_ra.copy(), data_dec.copy()]).transpose(), fmt='%-0.4f')
    np.savetxt(default_rand, np.array([rand_ra.copy(), rand_dec.copy()]).transpose(), fmt='%-0.4f')
    print ('Temporary data and random ascii files created')
    sys.stdout.flush()
    
    # Create the parameters file
    time_params = init_kdtpcf_params(param_path, s_min, s_max, nbins, log_bins, default_data, default_rand, default_out, jk_reg, nthr)
    print ('Parameter file created')
    sys.stdout.flush()
    
    ## Function to print time
    def print_time(time) :
        h = int(time//3600)
        m = int((time - 3600.0*h)//60)
        s = time - 3600.0*h - 60.0*m
        print ('{:>02d}:{:>02d}:{:>04.2f}'.format(h, m, s))
        sys.stdout.flush()
        return ()
    
    # Run KDTPCF
    os.chdir('/home/wagoner47/autocorrelation_code/KDTPCF')
    print ('Current working directory: {}'.format(os.getcwd()))
    sys.stdout.flush()
    # Save the command line arguments for the subprocess
    if os.path.dirname(param_path) == os.getcwd() :
        cmdarg = ["./kdtpcf", "{}".format(os.path.basename(param_path))]
    else :
        cmdarg = ["./kdtpcf", "{}".format(param_path)]
    ## Debugging: print the command that is being fed to subprocess.Popen
    print (subprocess.list2cmdline(cmdarg))
    sys.stdout.flush()
    # Call subprocess.Popen to run the executable
    kp = subprocess.Popen(cmdarg, stdout=subprocess.PIPE)
    for line in iter(kp.stdout.readline, b"") :
        print (line)
        sys.stdout.flush()
    # Periodically print the time while the subprocess is running so I know things are still working
#     kdtpcf_start = time.time()
#     while kp.poll() == None :
#         print_time(time.time() - kdtpcf_start)
#         time.sleep(29.97)
    os.chdir('/home/wagoner47/autocorrelation_code/2d_autocorrelation')
    print ('Current working directory: {}'.format(os.getcwd()))
    sys.stdout.flush()
    print ('Correlation code called')
    sys.stdout.flush()
    
    
    # Get the results from the files
    results, time_read = read_output(default_out, jk_reg)
    print ('Output collected')
    sys.stdout.flush()
    
    # Delete the old output files and data and random files
    os.remove(default_data)
    os.remove(default_rand)
    os.remove(default_out + '_ddbins')
    os.remove(default_out + '_drbins')
    os.remove(default_out + '_rdbins')
    os.remove(default_out + '_rrbins')
    os.rmdir(temp_path)
    print ('Temporary files deleted')
    sys.stdout.flush()
    ## Debugging: don't delete temporary files so that I can verify correct usage.
#     print ('Temporary files located at:\n\t{:s}'.format(temp_path))
#     print ('Data ASCII file:\n\t{:s}'.format(default_data))
#     print ('Random ASCII file:\n\t{:s}'.format(default_rand))
#     print ('DD output file:\n\t{:s}'.format(default_out + '_ddbins'))
    print ('Parameter file:\n\t{:s}'.format(param_path))
    sys.stdout.flush()
    
    # Stop timing
    end_time = time.time()
    
    # Return objects
    return (results, end_time - start_time)


def calculate_correlation_kdtpcf(results, jk_reg=4) :
    # Import python modules
    import time
    import numpy as np
    import sys
    
    # Start timing
    start_time = time.time()
    
    # Calculate number of centers from jk_reg
    ncen = 2**jk_reg
    
    # Get out the paircounts for the full sample
    ## DD
    dd = results['dd']['npairs'].copy()
    ## DR
    dr = results['dr']['npairs'].copy()
    ## RD
    rd = results['rd']['npairs'].copy()
    ## RR
    rr = results['rr']['npairs'].copy()
    
    # Calculate correlation function for full sample
    w = (dd.copy() - dr.copy() - rd.copy() + rr.copy())/rr.copy()
    del dd, dr, rd, rr
    print ('Full correlation function calculated')
    sys.stdout.flush()
    
    # Get out jackknife paircounts and find errors
    sig_sq = np.zeros_like(w)
    for i in range(ncen) :
        dd_reg = results['dd'][i]['npairs'].copy()
        dr_reg = results['dr'][i]['npairs'].copy()
        rd_reg = results['rd'][i]['npairs'].copy()
        rr_reg = results['rr'][i]['npairs'].copy()
        w_reg = (dd_reg.copy() - dr_reg.copy() - rd_reg.copy() + rr_reg.copy())/rr_reg.copy()
        sig_sq += (w.copy() - w_reg.copy())**(2)
        del dd_reg, dr_reg, rd_reg, rr_reg, w_reg
    sig_sq = sig_sq*(float(ncen - 1)/float(ncen))
    errs = (sig_sq.copy())**(0.5)
    del sig_sq
    print ('Errors calculated')
    sys.stdout.flush()
    
    # Stop timing
    end_time = time.time()
    
    # Return objects
    return (w, errs, end_time - start_time)

