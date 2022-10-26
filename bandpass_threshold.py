# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:02:28 2018

@author: aaron

# bandpass_plot.py - Plot the bandpass for 17a085 (K-band).

Modified by rsw on 20 Jan 2022
"""

import matplotlib
#matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import multiprocessing as mp

from subprocess import call

def run_bp(infile, bpfile):
    bp_cmd = "bandpass %s > %s" %(infile, bpfile)
    print(bp_cmd)
    call(bp_cmd, shell=True)
    return 

def calc_bandpass(infiles, workdir):
    """
    Calculate the bandpasses for a list of input files 
    using the SIGPROC taksk bandpass

    Will output bandpass for each file as a text file 
    with the same base as infiles but with "bpass" extension 

    This will calculate the bandpass by averaging over all 
    the full observation.  We may want to change this later 
    to do it in segments or something to avoid having to 
    flag a whole channel.

    Return list of bp_files
    """
     
    bp_files = []
    tstart = time.time()
    jobs = []
    for infile in infiles:
        infname = infile.rsplit('/')[-1] 
        inbase  = infname.rsplit('.', 1)[0]
        bpfile = "%s/%s.bpass" %(workdir, inbase)
        bp_files.append(bpfile)
        p = mp.Process(target=run_bp, args=(infile, bpfile))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    tstop = time.time()
    print("Took %.1f minutes" %( (tstop-tstart)/60.))
   
    return bp_files


def calc_bandpass_old(infiles, workdir):
    """
    Calculate the bandpasses for a list of input files 
    using the SIGPROC taksk bandpass

    Will output bandpass for each file as a text file 
    with the same base as infiles but with "bpass" extension 

    This will calculate the bandpass by averaging over all 
    the full observation.  We may want to change this later 
    to do it in segments or something to avoid having to 
    flag a whole channel.

    Return list of bp_files
    """
     
    bp_files = []
    for infile in infiles:
        tstart = time.time()
        infname = infile.rsplit('/')[-1] 
        inbase  = infname.rsplit('.', 1)[0]
        bpfile = "%s/%s.bpass" %(workdir, inbase)
        
        bp_cmd = "bandpass %s > %s" %(infile, bpfile)
        print(bp_cmd)
        call(bp_cmd, shell=True)

        bp_files.append(bpfile)
        tstop = time.time()
        print("Took %.1f minutes" %( (tstop-tstart)/60.))

    return bp_files
        

def moving_median(data, window):
    """
    Calculate running median and stdev
    """
    startIdxOffset = np.floor(np.divide(window, 2.0))
    endIdxOffset = np.ceil(np.divide(window, 2.0))
    
    startIndex = startIdxOffset
    endIndex = len(data) - 1 - endIdxOffset
    
    halfWindow = 0.0
    
    if (np.mod(window, 2.0) == 0):
        halfWindow = int(np.divide(window, 2.0))
    else:
        halfWindow = int(np.divide(window - 1.0, 2.0))
    
    mov_median = np.zeros(len(data))
    mov_std = np.zeros(len(data))

    startIndex = int(startIndex)
    endIndex = int(endIndex)

    # Calculate the moving median and std. dev. associated 
    # with each interval.
    for i in np.arange(startIndex, endIndex + 1, 1):
        istart = int(i - halfWindow)
        istop  = int(i + halfWindow + 1)
        medianValue = np.median(data[istart : istop])
        stdValue = np.std(data[istart : istop], ddof=1)

        mov_median[i] = medianValue
        mov_std[i] = stdValue
    
    # Set the values at the end points.
    for i in np.arange(0, startIndex, 1):
        mov_median[i] = mov_median[startIndex]
        mov_std[i] = mov_std[startIndex]
    
    for i in np.arange(endIndex + 1, len(data), 1):
        mov_median[i] = mov_median[endIndex]
        mov_std[i] = mov_std[endIndex]

    return mov_median, mov_std;
    

def read_bp(bp_filename):
    """
    Read the bandpass file created by SIGPROC bandpass
    """
    # Read data from the file.
    freqs = []
    bp = []
    with open(bp_filename, 'r') as fin:
        for line in fin:
            if line[0] in [" ", "\n"]:
                continue
            else: pass
            cols = line.split()
            freq_val = float(cols[0])
            bp_val = float(cols[1])
            
            freqs.append(freq_val)
            bp.append(bp_val)

    freqs = np.array(freqs)
    bp = np.array(bp)

    return freqs, bp


def plot_bp(freqs, bp, mask_chans, diff_thresh=None, 
            val_thresh=None, outfile=None):
    """
    Plot bandpass with masked chans indicated
    """
    chans = np.arange(0, len(freqs), 1)
    good_chans = np.setdiff1d(chans, mask_chans)

    # if outputting file, turn off interactive mode
    if outfile is not None:
        plt.ioff()
    else:
        plt.ion()
 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(freqs[good_chans], bp[good_chans], 'k.')
    ax.plot(freqs[mask_chans], bp[mask_chans], ls='', 
            marker='o', mec='r', mfc='none')

    if val_thresh is not None:
        ax.axhline(y=val_thresh, ls='--', c='g', alpha=0.5)
    else: pass

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("BP Coeff")    

    title_str = ""
    if diff_thresh is not None:
        diff_str = "diff_thresh = %.2f" %(diff_thresh)
        title_str += diff_str 
        if val_thresh is not None:
            title_str += ", "
        else: pass
    if val_thresh is not None:
        val_str = "val_thresh = %.2f" %(val_thresh)
        title_str += val_str
    else: pass

    if len(title_str):
        ax.set_title(title_str)
    else: pass

    ax.set_yscale('log')
   
    # If outfile, then save file, close window, and 
    # turn interactive mode back on
    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
    else: 
        plt.show()

    return

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def del_chans_to_string(nums):
    """
    Take list of channel numbers to remove and convert 
    them to a string that can be input to PRESTO
    """
    # Get list of ranges 
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    ranges = list(zip(edges, edges))

    # Shorten string using ":" when necessary
    out_str = ""
    for i in np.arange(0, len(ranges), 1):
        if (ranges[i][0] == ranges[i][1]):
            out_str = out_str + str(ranges[i][0]) + ","
        else:
            out_str = out_str + str(ranges[i][0]) +\
                      ":" + str(ranges[i][1]) + ","

    # Remove trailing comma if nec
    if out_str[-1] == ',':
        out_str = out_str.rstrip(',')
    else: pass
    
    return out_str


def decimate_mask_chans(mask_chans, dec_fac):
    """
    Convert the full resolution mask channels to 
    apply for a decimated data set.

    mask_chans = full res mask channels 
    nchan = total number of channels in original data
    dec_fac = decimation factor
    """
    # dec_factor has to be an integer
    fac = int(dec_fac)

    if fac <=1:
        print("Don't need to do anything for fac=%d" %fac)
        return mask_chans
    else: pass

    dec_chans = np.unique( np.floor( mask_chans / dec_fac ) )
    dec_chans = dec_chans.astype('int')
    
    return dec_chans


def bp_filter(bp_file, diff_thresh=0.10, val_thresh=0.1, 
              nchan_win=32, outfile=None, dec_factor=1):
    """
    Run the filter on a single bandpass file and find 
    what channels need to be zapped

    diff_thresh = fractional diff threshold to mask chans 
                  Mask if abs((bp-med)/med) > diff_thresh
    
    val_thresh  = min value threshold to mask chans 
                  Mask if bp < val_thresh

    nchan_win = number of channels in moving window

    if outfile is specified, then save a plot showing 
    the bandpass and masked channels

    If dec_factor > 1, then return the channel indices for 
    when the channels are decimated by that factor
    """
    # Read in bp data 
    freqs, bp = read_bp(bp_file)

    # Calculate running median and stdev
    mov_median, mov_std = moving_median(bp, nchan_win)

    # Calc fractional difference from median
    # Fix in case there are any zeros
    abs_med = np.abs(mov_median)
    if np.any(abs_med):
        eps = 1e-3 * np.min( abs_med[ abs_med > 0 ] )
    else:
        eps = 1e-3  
    bp_diff = np.abs(bp - mov_median) / (abs_med + eps)

    # Find mask chans from diff
    diff_mask = np.where( bp_diff >= diff_thresh )[0]

    # Find mask chans from val
    val_mask = np.where( bp < val_thresh )[0]

    # Get unique, sorted list of all bad chans
    mask_chans = np.unique( np.hstack( (diff_mask, val_mask) ) )
    
    # Get list of good chans (might need)
    all_chans = np.arange(0, len(freqs))
    good_chans = np.setdiff1d(all_chans, mask_chans)

    # Make a plot if outfile is specified
    if outfile is not None:
        plot_bp(freqs, bp, mask_chans, diff_thresh=diff_thresh,
                val_thresh=val_thresh, outfile=outfile)
    else:
        pass 
   
    # If decimating, convert channels to new indices
    if dec_factor > 1:
        mask_chans = decimate_mask_chans(mask_chans, dec_factor)
    else: 
        pass
 
    # Convert mask_chan array to string
    mask_chan_str = del_chans_to_string(mask_chans)

    return mask_chan_str


def get_zapchan_strings(bp_files, diff_thresh=0.10, val_thresh=0.1,
                        nchan_win=32, dec_factor=1, saveplots=True):
    """
    Run the bandpass filtering on each of the bandpass 
    files in the bp_files list using the parameters specified 
    by diff_thresh, val_thresh, and nchan_win

    If saveplots = True, then a plot of the bandpass showing
    what channels were flagged will be made

    They will be saved to [bp_file_base]_bpmask.png

    This function returns a list of zapchan strings that 
    can be used for flagging 
    """
    zap_chan_strs = []
    for bpfile in bp_files:
        outfile = "%s_bp_zap.png" %(bpfile.rstrip('.bpass'))
        mstr = bp_filter(bpfile, diff_thresh=diff_thresh, 
                         val_thresh=val_thresh, nchan_win=nchan_win, 
                         dec_factor=dec_factor, outfile=outfile)
        zap_chan_strs.append(mstr)
    
    return zap_chan_strs
