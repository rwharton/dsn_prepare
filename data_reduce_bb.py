"""
Created on Tue Sep 4 16:00:45 2018

@author: pearlman

data_reduce.py

Modified on 18 Jan 2022 by rsw
"""

import getopt, sys
import copy
import os
import time
import glob 
import shutil

from subprocess import call, check_call, check_output, Popen

import numpy as np

import filterbank

import h5py
from scipy import signal
from threading import Thread

import data_reduce_params_bb as par
import bandpass_threshold as bp_zap

def get_inbase(infile):
    """
    Strip off suffix to get basename for a file
    """
    inbase = infile.rsplit('.', 1)[0]
    return inbase


def filename_from_path(inpath):
    """
    split path to get filename
    """
    fname = inpath.split('/')[-1]
    return fname


def get_filename(basename, subnum, pol):
    """
    Generate the input file name from basename, 
    suband number, and polarization

    This may need to be changed if convention changes
    """
    fname = "%s_K%d-%s.fil" %(basename, subnum, pol)
    return fname


def freq_sort_files(infiles):
    """
    Sort file list in decreasing frequency order
    """
    # Get first chan freqs for each file
    freq_list = []
    for infile in infiles:
        hdr = filterbank.read_header(infile)
        fch1 = hdr[0]['fch1']
        freq_list.append(fch1)

    # Sort in descending order
    idx_sort = np.argsort(freq_list)[::-1]

    # Re-order
    sortfiles = []
    for idx in idx_sort:
        sortfiles.append(infiles[idx])

    return sortfiles
    

def check_input_file():
    """
    make sure input file exists
    """
    indir    = par.indir 
    infile   = par.infile
    inpath   = "%s/%s" %(indir, infile)

    if not os.path.exists(inpath):
        print("Data file not found:")
        print("  %s" %inpath)
        sys.exit(0)
    else: 
        print("Found file:")
        print("  %s" %inpath)

    return inpath


def bandpass_zapchans(infiles):
    """
    Calculate bandpass using SIGPROC bandpass

    Using those bandpasses, do a median deviation 
    filter to identify bad channels

    return list of zapchan strings for zapping

    if data will be decimated later, the channel indices 
    will be converted for use in the decimated data set

    will also produce plots if desired
    """
    tstart = time.time()

    workdir = par.workdir
    diff_thresh = par.diff_thresh
    val_thresh  = par.val_thresh 
    nchan_win   = par.nchan_win 
    dec_factor  = par.chan_dec_factor

    # Generate bpass files (takes a while)
    bp_files = bp_zap.calc_bandpass(infiles, workdir)

    # Get list of zapchan strings 
    zap_chans = bp_zap.get_zapchan_strings(bp_files, 
                                  diff_thresh=diff_thresh,
                                  val_thresh=val_thresh,
                                  nchan_win=nchan_win, 
                                  dec_factor=dec_factor, 
                                  saveplots=True)

    tstop = time.time()
    tdur = tstop - tstart

    return zap_chans, tdur


def decimate_chans(infiles):
    """
    Decimate infiles by factor specified in par file

    Return output files and time
    """
    tstart = time.time()

    workdir = par.workdir
    dec_fac = par.chan_dec_factor 

    outfiles = []
    for infile in infiles:
        fin_name  = infile.rsplit('/')[-1]
        fin_base  = fin_name.rsplit('.')[0]
        
        dec_file = "%s/%s_dec%d.corr" %(workdir, fin_base, dec_fac)

        # Do the actual decimation
        dec_cmd = "decimate -c %d %s > %s" %(dec_fac, infile, dec_file)
        print(dec_cmd)
        call(dec_cmd, shell=True)

        # Call splice_fb_nobary again to fix the bug in decimate, 
        # so that the header of the filterbank is pulsarcentric.
        tmp_file = "%s/%s_dec%d-tmp.corr" %(workdir, fin_base, dec_fac)
        splice_cmd = "splice_fb_nobary %s > %s" %(dec_file, tmp_file)
        print(splice_cmd)
        call(splice_cmd, shell=True)

        # Now rename tmp file to dec file
        mv_cmd = "mv %s %s" %(tmp_file, dec_file)
        print(mv_cmd)
        call(mv_cmd, shell=True)

        outfiles.append(dec_file)

    tstop = time.time()
    tdur = tstop - tstart

    return outfiles, tdur


def fix_chan_header(infiles):
    """
    Fix incorrect channel width in header
        
    Will account for decimation
    """
    tstart = time.time()

    dec_fac = par.chan_dec_factor 
    chan_df = par.chan_df
    new_df = dec_fac * chan_df
    
    for infile in infiles:
        edit_cmd = "filedit -F %.8f %s" %(new_df, infile) 
        print(edit_cmd)
        call(edit_cmd, shell=True)
   
    tstop = time.time()
    tdur = tstop - tstart 

    return infiles, tdur


def zap_channels(infiles, zapchans):
    """
    Zap bad channels as determined from bandpass_zapchans

    infiles is list of data files to zap
    zapchans is list of strings of channels to zap

    return list of zapped files, which will be in the same 
    place as input files but with "_zap" appended to the    
    base file name
    """
    tstart = time.time()

    src_dir = par.src_dir
    outfiles = []

    for ii, infile in enumerate(infiles):
        inbase = get_inbase(infile)
        outfile = "%s_zap.corr" %(inbase)
        zap_str = zapchans[ii]
    
        zap_cmd = "python " +\
                  "%s/m_fb_zapchan.py " %src_dir +\
                     "--inputFilename %s " %infile +\
                     "--outputFilename %s " %outfile +\
                     "--zapChan %s " %zap_str +\
                     "--clean"

        print(zap_cmd)
        call(zap_cmd, shell=True)

        outfiles.append(outfile)

    tstop = time.time()
    tdur = tstop - tstart 

    return outfiles, tdur
        

def combine_subbands(infiles):
    """
    Combine all the subbands into one file

    Files should already be in descending freq order
    """
    tstart = time.time()

    workdir = par.workdir
    outbase = par.outbase
    dec_factor = par.chan_dec_factor
    outfile = "%s/%s_dec%d.corr" %(workdir, outbase, dec_factor)

    infiles_str = ""
    for infile in infiles:
        infiles_str += infile
        infiles_str += " " 
    
    splice_cmd = "splice_fb_nobary %s > %s" %(infiles_str, outfile)
    print(splice_cmd)
    call(splice_cmd, shell=True)

    tstop = time.time()
    tdur = tstop - tstart

    return outfile, tdur


def get_comma_strings(f0, nharm, width):
    """
    Turn lists into comma separated strings
    """
    f0_str = ""
    nh_str = ""
    ww_str = ""
    N = len(f0)
    for ii in range(N):
        f0_str += "%.3f," %(f0[ii])
        nh_str += "%d," %(nharm[ii])
        ww_str += "%.1f," %(width[ii])

    f0_str = f0_str[:-1]
    nh_str = nh_str[:-1]
    ww_str = ww_str[:-1]

    return f0_str, nh_str, ww_str


def filter_freqs(infile, f0, nharm, width, outfile=None):
    """
    Filter out RFI signals and harmonics
    """
    tstart = time.time()

    src_dir = par.src_dir
    nproc   = par.filter_nproc

    if outfile is None:
        inbase = get_inbase(infile)
        outfile = "%s_filter.corr" %(inbase)
    else: pass

    f0_str, nh_str, ww_str = get_comma_strings(f0, nharm, width)
    
    filter_cmd = "python " +\
                 "%s/m_fb_freq_filter_parallel_new.py " %src_dir +\
                 "--inputFilename %s " %infile +\
                 "--outputFilename %s " %outfile +\
                 "--f0 %s " %f0_str +\
                 "--nharm %s " %nh_str +\
                 "--width %s " %ww_str +\
                 "--numProcessors %d " %nproc +\
                 "--clean"
    
    print(filter_cmd)
    call(filter_cmd, shell=True)

    tstop = time.time()
    tdur = tstop - tstart 

    return outfile, tdur


def bandpass(infile):
    """
    Bandpass correct data

    Run rfifind on corrected data

    Bandpass correct again after rfi masking
    """
    tstart = time.time()

    # First we do a bandpass on input data
    bpass_time = par.bpass_tmin 
    ra_str     = par.ra_str
    dec_str    = par.dec_str 

    workdir = par.workdir
   
    infile_name = filename_from_path(infile) 
    inbase = get_inbase(infile_name)
    outbase = "%s/%s" %(workdir, inbase)
    bfile1 = "%s_bp.corr" %(outbase)

    bp1_cmd = "prepfil " +\
              "--bandpass=%.2f " %bpass_time +\
              "--ra=%s " %ra_str +\
              "--dec=%s " %dec_str +\
              "%s %s" %(infile, bfile1)
    print(bp1_cmd)
    call(bp1_cmd, shell=True) 

    # Next we run rfifind
    rfi_time     = par.rfi_time
    rfi_chanfrac = par.rfi_chanfrac 
    rfi_clip1    = par.rfi_clip1 
    rfi_freqsig  = par.rfi_freqsig 

    rfi_base = "%s_bp" %outbase 
    
    rfi_cmd = "rfifind " +\
              "-time %.2f " %rfi_time +\
              "-chanfrac %.2f " %rfi_chanfrac +\
              "-clip %.2f " %rfi_clip1 +\
              "-freqsig %.2f " %rfi_freqsig +\
              "-filterbank %s " %bfile1 +\
              "-o %s " %rfi_base
    print(rfi_cmd)
    call(rfi_cmd, shell=True)

    # Now we run bandpass again with the rfi mask 
    bfile2 = "%s_bp_rfi.corr" %(outbase)
    mask_file  = "%s_rfifind.mask" %rfi_base

    bp2_cmd = "prepfil " +\
              "--bandpass=%.2f " %bpass_time +\
              "--ra=%s " %ra_str +\
              "--dec=%s " %dec_str +\
              "--mask %s " %mask_file +\
              "%s %s" %(infile, bfile2)
    print(bp2_cmd)
    call(bp2_cmd, shell=True) 

    # If it worked, delete intermediatary file
    if os.path.exists(bfile2):
        os.remove(bfile1)
    else: pass

    tstop = time.time()
    tdur = tstop - tstart 
    
    return bfile2, tdur


def filter_avg(infile):
    """
    Run moving average filter, then rfifind, then apply mask to data
    """
    tstart = time.time()

    src_dir = par.src_dir

    # First we run the moving average filter
    avg_tconst = par.avg_filter_timeconst
    avg_nproc  = par.avg_filter_nproc

    inbase = infile.rstrip(".corr")
    avg1_file = "%s_avg.corr" %inbase
    
    avg_cmd = "python " +\
              "%s/m_fb_filter_norm_jit.py " %src_dir +\
              "--inputFilename %s " %infile +\
              "--outputFilename %s " %avg1_file +\
              "--timeConstLong %.2f " %avg_tconst +\
              "--numProcessors %d " %avg_nproc +\
              "--clean "
    print(avg_cmd)
    call(avg_cmd, shell=True)
    
    # Next we run rfifind
    rfi_time     = par.rfi_time
    rfi_chanfrac = par.rfi_chanfrac 
    rfi_clip2    = par.rfi_clip2 
    rfi_freqsig  = par.rfi_freqsig 
    
    rfi_base = "%s_avg" %inbase

    rfi_cmd = "rfifind " +\
              "-time %.2f " %rfi_time +\
              "-chanfrac %.2f " %rfi_chanfrac +\
              "-freqsig %.2f " %rfi_freqsig +\
              "-filterbank %s " %avg1_file +\
              "-o %s " %rfi_base
    print(rfi_cmd)
    call(rfi_cmd, shell=True)

    # Now apply mask with prepfil
    rfi_mask = "%s_rfifind.mask" %rfi_base
    avg2_file = "%s_avg_masked.corr" %inbase

    prep_cmd = "prepfil " +\
               "--mask=%s " %rfi_mask +\
               "%s %s " %(avg1_file, avg2_file) 
    print(prep_cmd)
    call(prep_cmd, shell=True)

    # If successful, remove intermediary file
    if os.path.exists(avg2_file):
        os.remove(avg1_file)
    else: pass

    tstop = time.time()
    tdur = tstop - tstart 
    
    return avg2_file, tdur


def rename_output_file(infile):
    """
    Link infile to clear output name
    """
    outdir  = par.outdir
    outbase = par.outbase
    outfile = "%s/%s_final.corr" %(outdir, outbase)

    # Check that target file doesnt already exist
    if os.path.isfile(outfile):
        print("File exists:  %s" %outfile)
        print("Skipping final link...")
        return 
    else: pass

    mv_cmd = "mv %s %s" %(infile, outfile)
    print(mv_cmd)
    call(mv_cmd, shell=True)
    
    return


def delete_files(flist):
    """
    Delete files in list flist
    """
    for dfile in flist:
        if os.path.exists(dfile):
            os.remove(dfile)
        else:
            continue
    return


def organize_output():
    """
    organize output files into folders

    NOTE: We are assuming everything already in output dir
    """
    top_dir = par.outdir 

    # Place for rfifind masks
    mask_dir = "%s/masks" %(top_dir)
    os.mkdir(mask_dir)
    
    # Place for bandpass files
    bpass_dir = "%s/bpass" %(top_dir)
    os.mkdir(bpass_dir)

    # Get rfifind files and move to mask
    rfi_files = glob.glob("%s/*rfifind*"%(top_dir))
    for rfi_file in rfi_files:
        shutil.move(rfi_file, mask_dir)

    # Get *.bpass files and move to bpass
    bpass_files = glob.glob('%s/*.bpass' %(top_dir))
    for bpass_file in bpass_files:
        shutil.move(bpass_file, bpass_dir)
    
    # Get *.bandpass files and move to bpass
    bpass2_files = glob.glob('%s/*.bandpass' %(top_dir))
    for bpass2_file in bpass2_files:
        shutil.move(bpass2_file, bpass_dir)

    # Move plots to bpass
    bpng_files = glob.glob('%s/*bp_zap.png' %(top_dir))
    for bpng_file in bpng_files:
        shutil.move(bpng_file, bpass_dir)

    ## Copy par file to output
    #par_file = par.par_file 
    #if par.copy_par:
    #    shutil.copy(par_file, top_dir)

    return

            


def main():
    """
    Call all of the processing steps
    
    Info will come from param file
    
    Time the non trivial steps
    """
    tstart = time.time()

    # Get file paths based on param file
    infile = check_input_file()

    # Copy par file to out dir
    if par.copy_par:
        shutil.copy(par.par_file, par.outdir)

    # Bandpass + RFI + Bandpass
    bpass_file, bpass_time = bandpass(infile)

    # Running avg baseline filter
    bpass_bl_file, avg_time = filter_avg(bpass_file)
    
    # Rename final cal file to cleaner output name
    rename_output_file(bpass_bl_file)

    # Organize output
    organize_output()

    tstop = time.time()
    total_time = tstop - tstart

    # Now print summary of times
    print("##################################################")
    print("##              TIME SUMMARY                    ##")
    print("##################################################")
    print("")    
    print("Bandpass:           %.1f minutes" %(bpass_time/60.))
    print("Baseline:           %.1f minutes" %(avg_time/60.))
    print("") 
    print("Total Time:         %.1f minutes" %(total_time/60.))
     
    return

debug = 0
    
if __name__ == "__main__":
    if debug:
        pass
    else:
        main()
