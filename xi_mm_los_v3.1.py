#! /usr/bin/env python

import os, sys
import numpy as np
from datetime import datetime
import argparse
from ConfigParser import ConfigParser
mod_path = os.path.expanduser('./my_modules')
if mod_path not in sys.path :
	sys.path.insert(0, mod_path)
import xi_mm_funcs_v5 as funcs
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
font_dict = {'family':'serif', 'serif':'cm', 'size':16}
rc('text', usetex=True)
rc('font', **font_dict)

# Command line options
parser = argparse.ArgumentParser(description='Calculate theoretical matter-matter \
correlation function with perturbations on positions')
## matter power directory location
parser.add_argument('matter_power_dir', help='Directory containing the matter power \
spectrum, k values, and z values from CosmoSIS')
## Cosmology config file
parser.add_argument('cosmology', help='Cosmology config file to convert to Mpc')
## save location
parser.add_argument('save_dir', help='Directory in which to save the results')
## min and max separation, and number of separation bins
parser.add_argument('--min_sep', type=float, default=60.0, help='Minimum separation bin \
for xi in Mpc (default: %(default)s)')
parser.add_argument('--max_sep', type=float, default=200.0, help='Maximum separation bin \
for xi in Mpc (default: %(default)s)')
parser.add_argument('--nbins', type=int, default=100, help='Number of \
separation bins for xi, assumes same number of bins in each direction (default: %(default)s)')
## sigma for position perturbations, for now only in z-direction
parser.add_argument('--sigmar', type=float, default=0.01, help='The error from the \
Gaussian for the perturbations in the LOS direction in Mpc. (default: %(default)s)')
## notification options
parser.add_argument('-t', action='store_true', help='If specified, print timing info')
parser.add_argument('--mail_options', default=None, help='Mail options config file. \
If not specified, no notification email will be sent at the end of execution (default: \
%(default)s)')
parser.add_argument('--nohup', help='nohup file to attach to notification email, if any')

plot_list = []

# Start timing
start = datetime.now()
# Read arguments
args = parser.parse_args()
mp_dir = os.path.normpath(args.matter_power_dir)
assert not os.path.exists(args.save_dir) or os.path.isdir(args.save_dir), 'Invalid \
option for save_dir: {}. Must be a directory!'.format(args.save_dir)
save_dir = os.path.normpath(args.save_dir)
cosmology = args.cosmology
min_sep = args.min_sep
max_sep = args.max_sep
nbins = args.nbins
sigma = args.sigmar
timing = args.t
## Load EMail options
if args.mail_options is not None :
	from noticeEMail import noticeEMail as mail
	mail_config = ConfigParser()
	mail_config.read(args.mail_options)
	mail_ops = mail_config._sections['mail_options']
## Make sure save directories exist
if not os.path.exists(save_dir) :
	os.makedirs(save_dir)
if not os.path.exists(os.path.join(save_dir, 'test_plots')) :
	os.makedirs(os.path.join(save_dir, 'test_plots'))


# Read matter power data
start_read = datetime.now()
try :
	kvals = np.loadtxt(os.path.join(mp_dir, 'k_h.txt'))
except IOError as e :
	if args.mail_options is not None :
		mail(start, mail_ops['usr'], mail_ops['psw'], mail_ops['fromaddr'], \
		mail_ops['toaddr'], exit=e, nohup=args.nohup, plots=[])
	sys.exit(e)
try :
	pvals = np.loadtxt(os.path.join(mp_dir, 'p_k.txt'))
except IOError as e :
	if args.mail_options is not None :
		mail(start, mail_ops['usr'], mail_ops['psw'], mail_ops['fromaddr'], \
		mail_ops['toaddr'], exit=e, nohup=args.nohup, plots=[])
	sys.exit(e)
try :
	zvals = np.loadtxt(os.path.join(mp_dir, 'z.txt'))
except IOError as e :
	if args.mail_options is not None :
		mail(start, mail_ops['usr'], mail_ops['psw'], mail_ops['fromaddr'], \
		mail_ops['toaddr'], exit=e, nohup=args.nohup, plots=[])
	sys.exit(e)
## Make sure the shape is correct
if zvals.size > 1 :
	if not pvals.shape == (zvals.size, kvals.size) :
		if not pvals.T.shape == (zvals.size, kvals.size) :
			print 'Error: matter power spectrum has different shape than z and k'
			print 'Power spectrum shape = {}, # z = {}, # k = {}'.format(pvals.shpae, \
			zvals.size, kvals.size)
			sys.stdout.flush()
			if args.mail_options is not None :
				mail(start, mail_ops['usr'], mail_ops['psw'], mail_ops['fromaddr'], \
				mail_ops['toaddr'], exit=1, nohup=args.nohup, plots=[])
			sys.exit(1)
		pvals = pvals.T
else :
	if not pvals.size == kvals.size :
		print 'Error: matter power spectrum has different size than k'
		print 'Power spectrum size = {}, # k = {}'.format(pvals.size, kvals.size)
		sys.stdout.flush()
		if args.mail_options is not None :
			mail(start, mail_ops['usr'], mail_ops['psw'], mail_ops['fromaddr'], \
			mail_ops['toaddr'], exit=1, nohup=args.nohup, plots=[])
		sys.exit(1)
## Print timing info if desired
if timing :
	print 'Time to read matter power data = {}'.format(datetime.now() - start_read)
sys.stdout.flush()
del start_read

# Read cosmology parameters
start_cosmo = datetime.now()
config = ConfigParser()
try :
	config.read(cosmology)
except IOError as e :
	if args.mail_options is not None :
		mail(start, mail_ops['usr'], mail_ops['psw'], mail_ops['fromaddr'], \
		mail_ops['toaddr'], exit=e, nohup=args.nohup, plots=[])
	sys.exit(e)
cosmo = config._sections['cosmological_parameters']
## Make sure values are floats
for key, value in cosmo.iteritems() :
	if key != '__name__' :
		cosmo[key] = float(value)
## Print timing info if desired
if timing :
	print 'Time to read cosmology parameters = {}'.format(datetime.now() - start_cosmo)
sys.stdout.flush()
del start_cosmo, config

# Convert k from h/Mpc to 1/Mpc
kvals *= cosmo['h0']
pvals /= np.power(cosmo['h0'], 3)
Pk = funcs.initialize_Pk(kvals, pvals)

## First step: let's get the min and max for r_\parallel
mu_grid1 = np.linspace(0., 1., num=5)
rp_min = min(10., min_sep)
rp_max = max(200., max_sep)

## Make plots of g(r_\parallel | mu_grid, rp_min/max)
ls = ['k-', 'b--', 'r-.', 'c:', 'm-']
labels = [r'$\mu = {}$'.format(mui) for mui in mu_grid1]

### Minimum
start_min = datetime.now()
print 'Finding minimum r_parallel'
sys.stdout.flush()
plt.figure(figsize=(10,6), facecolor='w')
plt.xlabel(r'$\pi$ (Mpc)')
plt.ylabel(r'$G(\mu, \pi, r_p = 10 \mathrm{Mpc})$')
### r_\parallel minimum: 0-400?
plt.title(r'$\pi_{min}$')
rl_min_grid = np.linspace(0, 50, num=101)
plt.xlim([0., 50.])
min_start = datetime.now()
gl_min_grid = funcs.g(mu_grid1, rl_min_grid, rp_min, Pk, sigma, flag=3)
if timing :
	print 'Time to compute grid for minimum r_parallel = '\
	'{}'.format(datetime.now() - min_start)
sys.stdout.flush()
del min_start
for i in range(mu_grid1.size) :
	plt.plot(rl_min_grid, gl_min_grid[i], ls[i], lw=2, label=labels[i])
plt.axvline(5, c='g')
plt.legend(loc=0, fontsize=font_dict['size'], fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'test_plots/rl_min.png'))
plot_list.append(os.path.join(save_dir, 'test_plots/rl_min.png'))
plt.close()
del rl_min_grid, gl_min_grid
if timing :
	print 'Time to make plot for minimum r_parallel = '\
	'{}'.format(datetime.now() - start_min)
sys.stdout.flush()
del start_min

### Maximum
start_max = datetime.now()
print 'Finding maximum r_parallel'
sys.stdout.flush()
plt.figure(figsize=(10,6), facecolor='w')
plt.xlabel(r'$\pi$ (Mpc)')
plt.ylabel(r'$G(\mu, \pi, r_p = 200 \mathrm{Mpc})$')
### r_\parallel maximum: ?
plt.title(r'$\pi_{max}$')
rl_max_grid = np.linspace(0., 4000., num=401)
plt.xlim([0., 4000.])
max_start = datetime.now()
gl_max_grid = funcs.g(mu_grid1, rl_max_grid, rp_max, Pk, sigma, flag=3)
if timing :
	print 'Time to compute grid for maximum r_parallel = '\
	'{}'.format(datetime.now() - max_start)
sys.stdout.flush()
del max_start
for i in range(mu_grid1.size) :
	plt.plot(rl_max_grid, gl_max_grid[i], ls[i], lw=2, label=labels[i])
plt.axvline(2000, c='g')
plt.legend(loc=0, fontsize=font_dict['size'], fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'test_plots/rl_max.png'))
plot_list.append(os.path.join(save_dir, 'test_plots/rl_max.png'))
plt.close()
del rl_max_grid, gl_max_grid
if timing :
	print 'Time to make plot for maximum r_parallel = '\
	'{}'.format(datetime.now() - start_max)
sys.stdout.flush()
del start_max

# Alert user
if args.mail_options is not None :
	mail(start, mail_ops['usr'], mail_ops['psw'], mail_ops['fromaddr'], \
	mail_ops['toaddr'], nohup=args.nohup, plots=plot_list)
else :
	print 'Total elapsed time = {}'.format(datetime.now() - start)
	sys.stdout.flush()