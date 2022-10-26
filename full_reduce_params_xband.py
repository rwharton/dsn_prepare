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
src_dir = '/src/prepare'
par_file = '%s/full_reduce_params_xband.py' %src_dir
copy_par = True
info_file = '%s/source_info.txt' %src_dir


#######################
##  Frequency Range  ##
#######################

# Truncate band to this freq range

lofreq = 8190.0
hifreq = 8550.0


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
avg_filter_timeconst = 15 # Pspin = 2
avg_filter_nproc     = 30


