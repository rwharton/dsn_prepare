# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:00:45 2017

@author: pearlman

Modified by rsw on 02 Feb 2022
"""

import getopt, sys
import copy
import os
from subprocess import call, check_call, check_output, Popen

import numpy as np
from numba import jit

import filterbank

import h5py
from scipy import signal
from threading import Thread

from fb_utils import readFilterbank, writeFilterbank

BLOCKSIZE = 1e6

@jit(nopython=True)
def movingAverage(data, window_size):
    """ 
    Compute the moving average using brute force numpy cumsum.
    jit to make it faster 
    """
    movingAvg = np.cumsum(data.astype(np.float64))
    movingAvg[window_size:] = movingAvg[window_size:] - movingAvg[:-window_size]
    movingAvg = movingAvg[window_size - 1:] / window_size

    # Prepend/append elements to the front/end of the list.
    appendLength = (len(data) - len(movingAvg)) / 2.0
    if (np.mod(appendLength, 1.0) != 0.0):
        prependArray = np.array([movingAvg[0]] * int(appendLength))
        appendArray = np.array([movingAvg[-1]] * int(appendLength + 1))

        movingAvg = np.concatenate((prependArray, movingAvg))
        movingAvg = np.concatenate((movingAvg, appendArray))
    else:
        prependArray = np.array([movingAvg[0]] * int(appendLength))
        appendArray = np.array([movingAvg[-1]] * int(appendLength))

        movingAvg = np.concatenate((prependArray, movingAvg))
        movingAvg = np.concatenate((movingAvg, appendArray))

    return movingAvg



def movingAvgData_Parallel(spectraData, inputHeader, timeConstant, 
                           numProcessors, logFile=""):
    """ 
    Remove variations in the data by calculating a moving average 
    in each channel of the filterbank file and subtracting this moving 
    average from the data. The window size of the moving average is 
    defined by the timeConstant input by the user. Modified to run 
    in parallel over multiple filterbank channels. 
    """
    
    if (logFile == ""):
        print("Detrending filterbank data... (timeConstant = %.3f s)\n" % timeConstant)
    else:
        logFile.write("Detrending filterbank data... (timeConstant = %.3f s)\n\n" % timeConstant)
    
    nsamples = []

    try:
        nsamples = float(inputHeader["nsamples"])
    except:
        inputHeader["nsamples"] = np.shape(spectraData)[1]
        nsamples = float(inputHeader["nsamples"])
    
    tsamp = float(inputHeader["tsamp"])
    nchans = float(inputHeader["nchans"])
    
    window = int(timeConstant / tsamp)
    
    if (window % 2 == 0):
        window = window + 1
    
    def worker(ichan):
        chanData = np.array(spectraData[ichan])
        movingAvg = movingAverage(chanData, window)
        chanData_detrend = np.subtract(chanData, movingAvg)
        spectraData[ichan] = chanData_detrend
    
    ichan = 0
    
    for iBatch in np.arange(0, int(nchans / numProcessors), 1):
        threads = []
        for iProcess in np.arange(0, numProcessors, 1):
            ichan = int(iProcess + (iBatch * numProcessors))
            
            t = Thread(target=worker, args=(ichan,))
            threads.append(t)
            
            progress = np.multiply(np.divide(ichan + 1.0, nchans), 100.0)
            
            if (logFile == ""):
                print("Detrending: Channel %i [%3.2f%%]" % (nchans - ichan, progress))
            else:
                logFile.write("Detrending: Channel %i [%3.2f%%]\n" % (nchans - ichan, progress))
        
        for x in threads:
            x.start()
        
        for x in threads:
            x.join()
    
    if (nchans % numProcessors):
        threads = []
        for ichan2 in np.arange(ichan + 1, int(nchans), 1):
            t = Thread(target=worker, args=(ichan2,))
            threads.append(t)
            
            progress = np.multiply(np.divide(ichan2 + 1.0, nchans), 100.0)
            
            if (logFile == ""):
                print("Detrending: Channel %i [%3.2f%%]" % (nchans - ichan2, progress))
            else:
                logFile.write("Detrending: Channel %i [%3.2f%%]\n" % (nchans - ichan2, progress))
        
        for x in threads:
            x.start()
        
        for x in threads:
            x.join()
    
    if (logFile == ""):
        print("\n")
    else:
        logFile.write("\n")
    
    return spectraData;


def zeroMean_Parallel(spectraData, inputHeader, numProcessors, 
                      logFile=""):
    """ 
    Make sure the time-series in each channel of the filterbank file 
    has a zero mean. Modified to run in parallel over multiple 
    filterbank channels. 
    """
    if (logFile == ""):
        print("Setting zero mean in filterbank data...\n")
    else:
        logFile.write("Setting zero mean in filterbank data...\n\n")
    
    nsamples = []

    try:
        nsamples = float(inputHeader["nsamples"])
    except:
        inputHeader["nsamples"] = np.shape(spectraData)[1]
        nsamples = float(inputHeader["nsamples"])
    
    tsamp = float(inputHeader["tsamp"])
    nchans = float(inputHeader["nchans"])
    
    def worker(ichan):
        chanData = spectraData[ichan]
        
        meanData = np.mean(chanData)
        stdData = np.std(chanData, ddof=1)
        
        if (stdData == 0.0):
            stdData = 1.0
         
        chanData = np.subtract(chanData, meanData)
        chanData = np.divide(chanData, stdData)
        
        spectraData[ichan] = chanData
    
    ichan = 0
    for iBatch in np.arange(0, int(nchans / numProcessors), 1):
        threads = []
        for iProcess in np.arange(0, numProcessors, 1):
            ichan = int(iProcess + (iBatch * numProcessors))
            
            t = Thread(target=worker, args=(ichan,))
            threads.append(t)
            
            progress = np.multiply(np.divide(ichan + 1.0, nchans), 100.0)
            
            if (logFile == ""):
                print("Zero Mean: Channel %i [%3.2f%%]" % (nchans - ichan, progress))
            else:
                logFile.write("Zero Mean: Channel %i [%3.2f%%]\n" % (nchans - ichan, progress))
        
        for x in threads:
            x.start()
        
        for x in threads:
            x.join()
    
    if (nchans % numProcessors):
        threads = []
        for ichan2 in np.arange(ichan + 1, int(nchans), 1):
            
            t = Thread(target=worker, args=(ichan2,))
            threads.append(t)
            
            progress = np.multiply(np.divide(ichan2 + 1.0, nchans), 100.0)
            if (logFile == ""):
                print("Zero Mean: Channel %i [%3.2f%%]" % (nchans - ichan2, progress))
            else:
                logFile.write("Zero Mean: Channel %i [%3.2f%%]\n" % (nchans - ichan2, progress))
        
        for x in threads:
            x.start()
        
        for x in threads:
            x.join()
    
    if (logFile == ""):
        print("\n")
    else:
        logFile.write("\n")
    
    return spectraData;



def usage():
    print("##################################")
    print("Aaron B. Pearlman")
    print("aaron.b.pearlman@caltech.edu")
    print("Division of Physics, Mathematics, and Astronomy")
    print("California Institute of Technology")
    print("Jet Propulsion Laboratory")
    print("##################################\n")

    print("""
    usage:  m_fb_filter.py [options]
        [-h, --help]                    : Display this help
        [--inputFilename]               : Name of input filterbank file
        [--outputFilename]              : Name of output filterbank file created after
                                          filtering is completed
        [--timeConstLong (SEC)]         : Length of time chunks for detrending long
                                          time-scale / low frequency variations
        [--timeConstShort (SEC)]        : Length of time chunk for detrending short
                                          time-scale / high frequency variations
                                          (designed for optimizing single pulse searches)
        [--numProcessors]               : Number of processors to be used for parallelizing
                                          the detrending algorithm. Default is 1. Must be
                                          less than the total number of channels.
        [--outputDir]                   : Output directory where the products of the data
                                          analysis will be stored
        [--logFile]                     : Name of the log file to store the data reduction
                                          output
        [--clean]                       : Flag to clean up intermediate reduction products.
                                          Default is FALSE
        
        
        This program reads a filterbank file, stores it in dynamic memory,
        and then removes variations from the data in each channel by subtracting
        a moving average. The mean of the data in each channel is then set to
        zero. The filtered data is written to a new filterbank file.

    """)



def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "inputFilename:outputFilename:timeConstLong:timeConstShort:numProcessors:outputDir:logFile:clean:",
                                   ["help", "inputFilename=", "outputFilename=",
                                    "timeConstLong=", "timeConstShort=",
                                    "numProcessors=", "outputDir=",
                                    "logFile=", "clean"])
    
    except getopt.GetoptError:
        # Print help information and exit.
        usage()
        sys.exit(2)
    
    if (len(sys.argv) == 1):
        usage()
        sys.exit(2)
    
    inputFilename=None
    outputFilename=None
    timeConstLong=None
    timeConstShort=None
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
        if o in ("--timeConstLong"):
            timeConstLong = a
        if o in ("--timeConstShort"):
            timeConstShort = a
        if o in ("--numProcessors"):
            numProcessors = a
        if o in ("--outputDir"):
            outputDir = a
        if o in ("--logFile"):
            logFile = a
        if o in ("--clean"):
            clean = True
    
    if ((inputFilename == None) | (outputFilename == None) \
        | (timeConstLong == None)):
        usage()
        sys.exit()
    
    if (timeConstLong != None):
        timeConstLong = float(timeConstLong)
    
    if (timeConstShort != None):
        timeConstShort = float(timeConstShort)
    
    if (numProcessors != None):
        numProcessors = float(numProcessors)
    
    if ((outputDir != None) & (logFile != None)):
        
        writeFile = open("%s/%s" % (outputDir, logFile), "w")
        
        spectraData, inputHeader, inputNbits, h5pyFile = readFilterbank(inputFilename, logFile=writeFile, BLOCKSIZE=BLOCKSIZE)
        
        if (numProcessors == None):
            numProcessors = 1
        
        spectraData = movingAvgData_Parallel(spectraData, inputHeader, timeConstLong,
                                             numProcessors, logFile=writeFile)
        
        if (timeConstShort != None):
            spectraData = movingAvgData_Parallel(spectraData, inputHeader, timeConstShort,
                                                 numProcessors, logFile=writeFile)
        
        spectraData = zeroMean_Parallel(spectraData, inputHeader, numProcessors,
                                        logFile=writeFile)
        
        writeFilterbank(outputFilename, spectraData, inputHeader, inputNbits,
                        logFile=writeFile, BLOCKSIZE=BLOCKSIZE)
        
        writeFile.close()
    
    else:
        
        spectraData, inputHeader, inputNbits, h5pyFile = readFilterbank(inputFilename, BLOCKSIZE=BLOCKSIZE)
        
        if (numProcessors == None):
            numProcessors = 1
        
        spectraData = movingAvgData_Parallel(spectraData, inputHeader, timeConstLong,
                                             numProcessors)
        
        if (timeConstShort != None):
            spectraData = movingAvgData_Parallel(spectraData, inputHeader, timeConstShort,
                                                 numProcessors)
        
        spectraData = zeroMean_Parallel(spectraData, inputHeader, numProcessors)
        
        writeFilterbank(outputFilename, spectraData, inputHeader, inputNbits, BLOCKSIZE=BLOCKSIZE)
    
    
    h5pyFile.close()
    
    if (clean == True):
        h5_path = "%s.hdf5" %(inputFilename)
        if os.path.exists(h5_path):
            print("Removing %s" %h5_path)
            os.remove(h5_path)
        else: pass
    else: pass

    
if __name__ == "__main__":
    main()
