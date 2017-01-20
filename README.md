# code_share2
Code for second group code share

## To run with default options
This call will reproduce the included test results. Simply replace **save_dir** with the directory in which to save the output.

python xi_mm_los_v3.1.py ./matter_power_nl default_cosmology.ini **save_dir**

## Other options
### These options are currently not used because the code isn't working that far yet
* `--min_sep`: Minimum separation for $\xi$, in Mpc (default: 60.0 Mpc)
* `--max_sep`: Maximum separation for $\xi$, in Mpc (default: 200.0 Mpc)
* `--nbins`: Number of separation bins for $\xi$. For $\xi(r_p, \uppi)$, assumes same number of bins in each direction (default: 100)
### These options are used currently
* `--sigmar`: The error for the Gaussian perturbations, in Mpc (default: 0.01 Mpc)
#### Note: I find the following options useful for tracking progress
* `-t`: If specified, print timing info
* `--mail_options`: Mail options config file. If specified, an email will be sent at the end of execution, assuming no errors cause the code to exit early (default: `None`)
* `--nohup`: Nohup file to which output is piped. If specified and notice email is being sent, nohup file will be attached (default: `None`)

## Timing info:
Time it took me to do the test run: 2:43:36.484742
