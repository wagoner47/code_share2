#! /usr/bin/env python

## Imports
import sys, os
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.integrate import quad
import numpy as np
from hankel import HankelTransform as HT

## Global spline variables that should be initialized to None or False
gmu_spl = None
grprl_spl = []
hspl = None
init_flag = False
## class to handle force flags for forcing grid evaluation for spline
class ForceFlagException(Exception) :
	pass

## Integration error limit
epsrel = 1.49e-8
## Spline error limit
spl_err = 1.e-4

## Power spectrum spline including large k fit
def initialize_Pk(kvals, pvals) :
	def line(x, a, b) :
		return a*x + b
	global Pspl, Pa, Pb
	Pspl = UnivariateSpline(kvals, pvals)
	## Fit at large k with a power law
	Pa, Pb = curve_fit(line, np.log(kvals[-4:]), np.log(pvals[-4:]))[0]
	## Fit at low k with a power law
	Pc, Pd = curve_fit(line, np.log(kvals[:4]), np.log(pvals[:4]))[0]
	def Pk(k) :
		if k > kvals[3] and k < kvals[-4] :
			return Pspl(k)
		elif k < kvals[3] :
			return np.exp(Pd)*np.power(k, Pc)
		else :
			return np.exp(Pb)*np.power(k, Pa)
	Pk = np.vectorize(Pk)
	return Pk

## Different forms for function f
# 1D Gaussian window function in the z direction
def f_1dgauss(x, kl, mu, r, Pk, sigmaz) :
	## Find Sqrt[1 - mu**2] to save writing space
	mup = np.sqrt(1. - np.power(mu, 2))
	## Try to find kp = x/(|r|*mup) for speed
	kp = x/(np.absolute(r)*mup)
	return np.nan_to_num(kp*Pk(np.sqrt(np.power(kl,2) + np.nan_to_num(np.power(kp,2)))))

# 1D Gaussian window function along the line of sight with CF(r_parallel, r_perp)
def f_los(x, mu, rl, rp, Pk, sigmar) :
	## Find Sqrt[1 - mu**2] to save writing space
	mup = np.sqrt(1. - np.power(mu, 2))
	## Do divisions for speed
	one_over_rp = np.nan_to_num(1./rp)
	one_over_mup = np.nan_to_num(1./mup)
	## If we get here, we can return the function of x for the Hankel transform
	return np.nan_to_num(np.cos(rl*one_over_rp*mu*one_over_mup*x)*np.exp(-np.power(\
	sigmar*one_over_rp*mu*one_over_mup*x, 2))*Pk(x*one_over_rp*one_over_mup)*\
	np.power(x, 2))
	

## Function f that calls the correct form of f given what we want -- pay attention to 
## what the arguments are in each case!
### 1 = 1D Gaussian in z: 
### 	arg1 = x = k_perp*|r|*\sqrt{1 - mu^2}: the variable for the Hankel transform
### 	arg2 = k_parallel
### 	arg3 = mu = \vec{r} \cdot \hat{z}
### 	arg4 = r
### 2 = 1D Gaussian in LOS direction, r_parallel and r_perp
### 	arg1 = x = k*r_\perp*\sqrt{1 - \mu^2}: the variable for the Hankel transform
### 	arg2 = mu = \vec{k} \cdot \hat{l}
### 	arg3 = r_parallel
### 	arg4 = r_perp
### 3 = 1D Gaussian in LOS direction, projected
### 	arg1 = x = k*r_\perp*\sqrt{1 - \mu^2}: the variable for the Hankel transform
### 	arg2 = mu = \vec{k} \cdot \hat{l}
### 	arg3 = r_parallel
### 	arg4 = r_perp
def f(arg1, arg2, arg3, arg4, Pk, sigma, flag=1) :
	if flag == 1 :
		## Return the integrand for the 1D Gaussian in the z direction
		return f_1dgauss(arg1, arg2, arg3, arg4, Pk, sigma)
	elif flag == 2 or flag == 3 :
		## Return the integrand for the 1D Gaussian in the LOS direction
		return f_los(arg1, arg2, arg3, arg4, Pk, sigma)
	
## f-integration function for vectorization: this version uses the hankel transform!
## Please refer to comments for definition of f to find the argument meanings
def f_hankel_integral(N, size, arg2, arg3, arg4, Pk, sigma, flag=1) :
	h = HT(nu=0, N=N, h=size)
	## Ensure things are arrays
	arg2 = np.atleast_1d(arg2)
	arg3 = np.atleast_1d(arg3)
	arg4 = np.atleast_1d(arg4)
	vals = np.empty((arg2.size, arg3.size, arg4.size), dtype=float)
	for i, a in zip(range(arg2.size), arg2) :
		## The loop over the first non-integrated argument
		for j, b in zip(range(arg3.size), arg3) :
			## The loop over the second non-integrated argument
			for k, c in zip(range(arg4.size), arg4) :
				## The loop over the third non-integrated argument
				### Lambda function for the Hankel transform
				y = lambda x: f(x, a, b, c, Pk, sigma, flag=flag)
				vals[i,j,k] = h.transform(y)[0]
				sys.stdout.flush()
				## Multiply by constants
				if flag == 1 :
					vals[i,j,k] *= 16.*np.power(np.pi, 2)*np.cos(a*b*c)*\
					np.exp(-np.power(a*sigma, 2))
				elif flag == 2 or flag == 3 :
					if vals[i,j,k] != 0. :
						## If the integral was 0, it is likely the constants will be infinite
						vals[i,j,k] *= (8.*np.power(np.pi, 2))/np.power(c*np.sqrt(1. - \
						np.power(a, 2)), 3)
				del y
	return vals

## G function
## Please refer to comments for definition of f to find the argument meanings for flag
### Note: eventually, this will use the splines if available
def g(arg2, arg3, arg4, Pk, sigma, flag=1) :
	if flag == 1 or flag == 2 or flag == 3 :
		return f_hankel_integral(500000, 0.0001, arg2, arg3, arg4, Pk, sigma, flag=flag)
		
## Initialize G spline: in progress, not ready yet
## Please refer to comments for definition of f to find the argument meanings
def initialize_grprl_spline(arg2, arg3, arg4, Pk, sigma, force=False, flag=1) :
	## Global variables
	global init_flag
	global grprl_spl
	## Set the init flag to True to force integration in g function
	init_flag = True
	try :
		## Raise an exception if force flag was given
		if force : raise ForceFlagException()
		## Try to read from file
		gtab = np.loadtxt('g_flag{}.txt'.format(flag)).reshape((np.atleast_1d(arg2).size, \
		np.atleast_1d(arg3).size, np.atleast_1d(arg4).size))
	except (ForceFlagException, IOError, UserWarning) :
		## Either force flag was given or the file doesn't exist yet. Evaluate and save.
		gtab = g(arg2, arg3, arg4, Pk, sigma, flag=flag)
		with file('g_flag{}.txt'.format(flag), 'w') as of :
			for slice in gtab :
				np.savetxt(of, slice, fmt='%-25.18f', footer='------')
	## Create the global spline object from the table
	### First do the splines in r_\perp, r_\parallel
	for i, a in zip(range(np.atleast_1d(arg2).size), np.atleast_1d(arg2)) :
		grprl_spl.append(RectBivariateSpline(arg3, arg4, gtab[i]))
	## Change the init flag back to False
	init_flag = False