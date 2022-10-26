# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:00:45 2017

@author: pearlman

m_fb_zapchan.py - Zero a list of channels input by the user.

Modified by rsw on 02 Feb 2022
"""

import getopt, sys
import copy
import os
from subprocess import call, check_call, check_output, Popen

import numpy as np

import filterbank

import h5py
from scipy import signal
from threading import Thread

from fb_utils import readFilterbank, writeFilterbank

BLOCKSIZE = 1e6

def zapChannels(fb_data, fb_zap_string):
    """ 
    Zero a list of channels in a filterbank file.
    Lowest channel index corresponds to the top of the 
    band/highest frequency. 
    """
    delimiter = ","
    zapList = fb_zap_string.split(delimiter)
    nchans, nsamples = np.shape(fb_data)
    
    for zapIndex in np.arange(0, len(zapList), 1):
        zapBand = zapList[zapIndex].split(":")
        
        if (len(zapBand) == 2):
            zapStart = int(zapBand[0])
            zapEnd = int(zapBand[1]) + 1
            nzapChannels = len(np.arange(zapStart, zapEnd, 1))
            fb_data[zapStart : zapEnd] = np.zeros((nzapChannels, nsamples))
        
        if (len(zapBand) == 1):
            zapChannel = int(zapBand[0])
            fb_data[zapChannel] = np.zeros((1, nsamples))
    
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
    usage:  m_fb_zapchan.py [options]
        [-h, --help]                    : Display this help
        [--inputFilename]               : Name of input filterbank file
        [--outputFilename]              : Name of output filterbank file created after
                                          filtering is completed
        [--zapChan]                     : String of channels to be zeroed/zapped,
                                          i.e. --zapChan 0:2,5,7:8
        [--outputDir]                   : Output directory where the products of the data
                                          analysis will be stored
        [--clean]                       : Flag to clean up intermediate reduction products.
                                          Default is FALSE
        
        
        This program reads a filterbank file, stores it in dynamic memory,
        and then zeroes a series of channels input by the user as a string.
        The channel limits in the string are inclusive, i.e., 0:2,5,7:8 will
        zero channels 0 through 2, channel 5, and channels 7 and 8.
        
        Index 0 is the top of the filterbank data structure, i.e. highest frequency.
        It is ordered in the following way:
        (rows) frequency [top -> high, bottom -> low],
        (columns) time [left -> start, right -> end]
        
        Example: 
    """)



def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "inputFilename:outputFilename:timeConstLong:timeConstShort:numProcessors:outputDir:logFile:clean:",
                                   ["help", "inputFilename=", "outputFilename=",
                                    "zapChan=", "outputDir=",
                                    "clean"])
    
    except getopt.GetoptError:
        # Print help information and exit.
        usage()
        sys.exit(2)
    
    if (len(sys.argv) == 1):
        usage()
        sys.exit(2)
    
    inputFilename=None
    outputFilename=None
    zapChan=None
    outputDir=None
    clean=None
    
    for o, a in opts:
        if (o in ("-h", "--help")):
            usage()
            sys.exit()
        
        if o in ("--inputFilename"):
            inputFilename = a
        if o in ("--outputFilename"):
            outputFilename = a
        if o in ("--zapChan"):
            zapChan = a
        if o in ("--outputDir"):
            outputDir = a
        if o in ("--clean"):
            clean = True
    
    
    if (outputDir != None):
        outputPathFilename = "%s/%s" % (outputDir, outputFilename)
        fb_data, fb_header, fb_Nbits, h5pyFile = readFilterbank(inputFilename, 
                                                           BLOCKSIZE=BLOCKSIZE)
        fb_data = zapChannels(fb_data, zapChan)
        writeFilterbank(outputPathFilename, fb_data, fb_header, fb_Nbits,
                        BLOCKSIZE=BLOCKSIZE)
        h5pyFile.close()
    else:
        fb_data, fb_header, fb_Nbits, h5pyFile = readFilterbank(inputFilename,
                                                          BLOCKSIZE=BLOCKSIZE)
        fb_data = zapChannels(fb_data, zapChan)
        writeFilterbank(outputFilename, fb_data, fb_header, fb_Nbits, 
                        BLOCKSIZE=BLOCKSIZE)
        h5pyFile.close()

    if (clean==True):
        h5_path = "%s.hdf5" %(inputFilename)
        if os.path.exists(h5_path):
            print("Removing %s" %h5_path)
            os.remove(h5_path)
        else: pass
    else: pass 
    
if __name__ == "__main__":
    main()
