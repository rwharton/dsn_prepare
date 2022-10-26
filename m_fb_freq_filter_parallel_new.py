# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:14:37 2018

@author: pearlman

Aaron B. Pearlman
aaron.b.pearlman@caltech.edu
Division of Physics, Mathematics, and Astronomy
California Institute of Technology
Jet Propoulsion Laboratory

Modified 31 Jan 2022 by rsw

m_fb_60hzfilter_parallel.py:

  Script for filtering out 60 Hz instrumental noise
  (and harmonics at 120 Hz and 180 Hz) from each
  channel of the filterbank file, with parallelization
  capabilities.

"""

import getopt, sys
from subprocess import call, check_call, check_output, Popen

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import h5py
import filterbank

from scipy import signal
from threading import Thread
from fb_utils import readFilterbank, writeFilterbank

BLOCKSIZE = 1e6

def butter_bandstop(nyq, cutoff_freq_start, cutoff_freq_stop, order=3):
    """ 
    Create a butterworth bandstop filter 
    (use a digital filter for real data!). 
    """
    cutoff_freq_start = cutoff_freq_start / nyq
    cutoff_freq_stop = cutoff_freq_stop / nyq
    
    b, a = signal.butter(order,
                        [cutoff_freq_start, cutoff_freq_stop], 
                        btype="bandstop", analog=False)
    
    return b, a


def get_freqs(f0, nharms, fnyq):
    """
    get array of freqs to zap
    """
    if f0 > fnyq:
        print("Fundamental above Nyquist")
        return np.array([])
    else: pass

    if nharms == -1:
        nh = int(fnyq / f0)
    else:
        nh = nharms

    freqs = f0 * (1 + np.arange(nh+1)) 

    print("Filter freqs (Hz):")
    for ff in freqs:
        print("  %0.3f" %ff)
    print("")

    return freqs


def fb_filter_harms(fb_data, fb_header, f0, nharms, width, 
                    numProcessors, logFile=""):
    """ 
    Filter out freq f0 and nharms harmonics.
    from the fb file.

    nharms = 0 : just filter fundamental (f0)
            -1 : filter all harmonics up to Nyquist
             N : filter f0, 2f0, ..., (N+1)f0
    
    Typical attenuation is ~120-150 dB around the filtered 
    frequencies. 
    """
    timeRes = float(fb_header["tsamp"])
    nsamples = np.shape(fb_data)[1]
    duration = np.divide(np.multiply(nsamples, timeRes), 3600.0)
    
    print("Time Resolution: %.6f s" % timeRes)
    print("nsamples: %i" % nsamples)
    print("Duration: %.2f hr\n" % duration)
    
    nchans = float(fb_header["nchans"])
    
    fs = 1.0 / timeRes
    nyq = 0.5 * fs
   
    freq_centers = get_freqs(f0, nharms, nyq)  
    freq_width = width
    freq_starts = freq_centers - freq_width
    freq_stops = freq_centers + freq_width
    
    # Apply filter to one channel of the filterbank file.
    def worker(ichan):
        fb_data[ichan] = signal.filtfilt(b, a, fb_data[ichan])
   
    # Apply to all 
    for iFilter in np.arange(0, len(freq_centers), 1):
        print("Filtering frequency: %.1f Hz" %(freq_centers[iFilter]))
        f_start = freq_starts[iFilter]
        f_stop  = freq_stops[iFilter] 
        b, a = butter_bandstop(nyq, f_start, f_stop, order=3)
        w, h = signal.freqz(b, a, worN=100000)
        
        # Need to parallelize filtering channel by channel.
        ichan = 0
        for iBatch in np.arange(0, int(nchans / numProcessors), 1):
            threads = []
            for iProcess in np.arange(0, numProcessors, 1):
                ichan = int(iProcess + (iBatch * numProcessors))
                
                t = Thread(target=worker, args=(ichan,))
                threads.append(t)
                
                progress = 100.0 * ((ichan + 1.0) / nchans) 
                
                if (logFile == ""):
                    print("Filtering [%.1f Hz]: Channel %i [%3.2f%%]" %(\
                          freq_centers[iFilter], nchans - ichan, progress))
                else:
                    logFile.write("Filtering [%.1f Hz]: Channel %i [%3.2f%%]\n" %(\
                          freq_centers[iFilter], nchans - ichan, progress))
            
            for x in threads:
                x.start()
            
            for x in threads:
                x.join()
        
        if (nchans % numProcessors):
            threads = []
            for ichan2 in np.arange(ichan + 1, int(nchans), 1):
                t = Thread(target=worker, args=(ichan2,))
                threads.append(t)
                
                progress = 100.0 * ((ichan + 1.0) / nchans)
                
                if (logFile == ""):
                    print("Filtering [%.1f Hz]: Channel %i [%3.2f%%]" %(\
                        freq_centers[iFilter], nchans - ichan2, progress))
                else:
                    logFile.write("Filtering [%.1f Hz]: Channel %i [%3.2f%%]\n" %(\
                           freq_centers[iFilter], nchans - ichan2, progress))
            
            for x in threads:
                x.start()
            
            for x in threads:
                x.join()
        
        if (logFile == ""):
            print("\n")
        else:
            logFile.write("\n")
        
    return fb_data;


def usage():
    print("##################################")
    print("Aaron B. Pearlman")
    print("aaron.b.pearlman@caltech.edu")
    print("Division of Physics, Mathematics, and Astronomy")
    print("California Institute of Technology")
    print("Jet Propulsion Laboratory")
    print("##################################\n")

    print("""
    usage:  m_fb_freq_filter_parallel.py [options]
     [-h, --help]            : Display this help
     [--inputFilename]       : Name of input filterbank file
     [--outputFilename]      : Name of output filterbank file created 
                               after filtering is completed
     [--f0]                  : Fundamental Freqs to Remove (Hz)
                               (comma separated list)
     [-nharm]                : Number of harmonics beyond each fundamental 
                               (comma separated list)
                               (0: only fundamental, -1: all up to Nyquist)
     [-width]                : Widths (in Hz) of filters
                               (comma separated list)
     [--numProcessors]       : Number of processors to be used for 
                               parallelizing the filtering algorithm. 
                               Default is 1. Must be less than the total 
                               number of channels.
     [--outputDir]           : Output directory where the products of 
                               the data
                               analysis will be stored.
     [--logFile]             : Name of the log file to store the 
                               data reduction output.
     [--clean]               : Flag to clean up intermediate reduction 
                               products.  Default is FALSE
     
     
     This program reads a filterbank file, stores it in dynamic memory,
     and then removes fundamental frequency f0 and nharm harmonics from 
     the filterbank file. The filtered data is written to a new 
     filterbank file.
        
    Example: m_fb_freq_filter_parallel.py --inputFilename input.corr --outputFilename output.corr --f0 60.0,50.0 --nharm 5,2 --width 1,5 --numProcessors 100 --outputDir /home/pearlman/fb_data/ --clean

    """)

def parse_list_string(opt_str, dtype):
    """
    Parse list of comma separated opt_str into array 
    with data type dtype ('int', 'float')
    """
    if dtype not in ["int", "float"]:
        print("dtype %s is unsupported" %(dtype))
        print("Must be \"int\" or \"float\"")
        return 
    else: pass
    
    str_list = opt_str.split(',')
    opt_arr = np.array(str_list, dtype=dtype)
    
    return opt_arr


def check_freq_opts(freqs, nharms, widths):
    """
    Make sure that the freqs / harms / widths 
    are compatible
    """
    Nf = len(freqs)
    Nh = len(nharms)
    Nw = len(widths)

    if (Nf == Nh == Nw ):
        return 1
    else:
        print("%d freqs, %d harms, %d widths given" %(Nf, Nh, Nw))
        print("Need to be the same number of each")
        return 0 
 

def main():
    try:
        opt_str = "inputFilename:" +\
                  "outputFilename:" +\
                  "f0:" +\
                  "nharm:" +\
                  "width:" +\
                  "numProcessors:" +\
                  "outputDir:" +\
                  "logFile:" +\
                  "clean:" 
        long_opts = ["help", "inputFilename=", "outputFilename=",
                     "f0=", "nharm=", "width=", "numProcessors=", 
                     "outputDir=", "logFile=", "clean"]
        opts, args = getopt.getopt(sys.argv[1:], opt_str, long_opts)
        print(opts)
    
    except getopt.GetoptError:
        # Print help information and exit.
        usage()
        sys.exit(2)
    
    if (len(sys.argv) == 1):
        usage()
        sys.exit(2)
    
    inputFilename=None
    outputFilename=None
    f0=None
    nharm=None
    width=None
    numProcessors=None
    outputDir=None
    logFile=None
    clean=None
    
    for o, a in opts:
        if (o in ("-h", "--help")):
            usage()
            sys.exit()
        
        if o in ("--inputFilename"):
            inputFilename = a
        if o in ("--outputFilename"):
            outputFilename = a
        if o in ("--f0"):
            f0 = a
        if o in ("--nharm"):
            nharm = a
        if o in ("--width"):
            width = a
        if o in ("--numProcessors"):
            numProcessors = a
        if o in ("--outputDir"):
            outputDir = a
        if o in ("--logFile"):
            logFile = a
        if o in ("--clean"):
            clean = True
    
    if ((inputFilename == None) | (outputFilename == None)):
        usage()
        sys.exit()
    
    if (numProcessors != None):
        numProcessors = float(numProcessors)
    else:
        numProcessors = 1
    
    if (f0 != None):
        f0 = parse_list_string(f0, 'float')
    else:
        f0 = np.array([ 60.0 ])
    
    if (nharm != None):
        nharm = parse_list_string(nharm, 'int')
    else:
        nharm = np.array([ 0 ])
    
    if (width != None):
        width = parse_list_string(width, 'float')
    else:
        width = np.array([ 1.0 ])

    # Check freqs / harms / widths
    if check_freq_opts(f0, nharm, width):
        pass
    else:
        sys.exit(2)

    Nsteps = len(f0)
    
    
    if ((outputDir != None) & (logFile != None)):
        writeFile = open("%s/%s" % (outputDir, logFile), "w")
        fb_data, fb_header, fb_Nbits, h5pyFile =\
                 readFilterbank(inputFilename, logFile=writeFile, 
                                BLOCKSIZE=BLOCKSIZE)
        
        for ii in range(Nsteps): 
            fb_data = fb_filter_harms(fb_data, fb_header, f0[ii], nharm[ii], 
                                      width[ii], numProcessors, logFile=writeFile)

        outputPath = "%s/%s" % (outputDir, outputFilename)
        
        if (outputDir == None):
             outputPath = outputFilename
        
        writeFilterbank(outputPath, fb_data, fb_header, fb_Nbits, 
                        logFile=writeFile, BLOCKSIZE=BLOCKSIZE)
        writeFile.close()
    else:
        fb_data, fb_header, fb_Nbits, h5pyFile =\
                 readFilterbank(inputFilename, BLOCKSIZE=BLOCKSIZE)

        for ii in range(Nsteps):        
            fb_data = fb_filter_harms(fb_data, fb_header, f0[ii], nharm[ii], 
                                      width[ii], numProcessors)

        outputPath = "%s/%s" % (outputDir, outputFilename)
        
        if (outputDir == None):
             outputPath = outputFilename
        
        writeFilterbank(outputPath, fb_data, fb_header, fb_Nbits, 
                        BLOCKSIZE=BLOCKSIZE)
    
    h5pyFile.close()
    
    if (clean == True):
        h5_path = "%s.hdf5" %(inputFilename)
        if os.path.exists(h5_path):
            print("Removing %s" %h5_path)
            os.remove(h5_path)
        else: pass
    else: pass


debug = 0
    
if __name__ == "__main__":
    if debug:
        pass
    else:
        main()
