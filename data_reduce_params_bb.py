######################################################################
##                        PARAMETER FILE                            ##
##                             for                                  ##
##                  data_reduce_params_bb.py                        ##
######################################################################

##########################
##  Processing Scripts  ##
##########################

# Directory containing all the scripts here
#   Note:  might have a different name if running in Docker  
src_dir = '/src/reduce'
par_file = '%s/data_reduce_params_bb.py' %src_dir
copy_par = True

#########################
##  Input/Output Data  ##
#########################

# Directory where data files are
#   Note:  might have a different name if running in Docker  
indir = '/output/s07-B1929+10'

# Input file name 
infile = 'B1929+10_trunc.fil'

# Base name for output data products
outbase = 'scan07-B1929'

# Directory where everything will go
#   Note:  might have a different name if running in Docker  
outdir = indir

###################
##  Working Dir  ##
###################

# Directory where we are doing work
# (For now just leave as outdir)
workdir = outdir


################
##  RA / DEC  ##
################

# RA and Dec strings needed for prepfil
# B1929+10
ra_str  = "193214.0570"
dec_str = "+105933.38"

# M77
#ra_str  = "024240.71"
#dec_str = "-000047.86"

###########################
##  Bandpass Correction  ##
###########################

# Time (minutes) to use for bpass solution
bpass_tmin = 5.0


###########################
##  rfifind RFI masking  ##
###########################

# rfifind paramters.  A "1" indicates the rfifind step 
# done during bandpass.  A "2" indicates the rfifind 
# done after the averaging filter.

rfi_time     = 0.0 
rfi_chanfrac = 0.1 
rfi_clip1    = 1.5  # before avg filter
rfi_clip2    = 0.0  # after avg filter
rfi_freqsig  = 16.0 


#############################
##  Moving Average Filter  ##
#############################

# Time const should be ~3 x Pspin
avg_filter_timeconst = 10 # Pspin = 2
avg_filter_nproc     = 30


