"""
Created on Mon Oct 16 16:00:45 2017

@author: pearlman

Modified on 12 Jan 2022 by rsw

Helper functions to read/write filterbank files
"""
import sys
import copy
import numpy as np
import filterbank
import h5py


def readFilterbank(inputFilename, logFile="", BLOCKSIZE = 1e6):
    """ 
    Read the filterbank file into memory. Store the data in a 
    dynamically accessible h5py file, stored in a binary .hdf5 file.
    """
    if (logFile == ""):
        print("Reading filterbank file (%s)...\n" % inputFilename)
    else:
        logFile.write("Reading filterbank file (%s)...\n\n" % inputFilename)
    
    fb = filterbank.FilterbankFile(inputFilename)
    
    inputHeader = copy.deepcopy(fb.header)
    inputNbits = fb.nbits
    totalChans = fb.nchans
    nchans = np.arange(0, fb.nchans-1, 1) # Top of the band is index 0.
    freqs = fb.frequencies
    startbin = 0
    endbin = fb.nspec
    nspec = np.subtract(endbin, startbin)
    nblocks = int(np.divide(nspec, BLOCKSIZE))
    remainder = nspec % BLOCKSIZE
    totalBlocks = nblocks
    
    if (remainder):
        totalBlocks = nblocks + 1
    
    h5pyFile = h5py.File("%s.hdf5" % inputFilename, "w")
    spectraData = h5pyFile.create_dataset("data", (totalChans, nspec), dtype="float32")
    
    for iblock in np.arange(0, nblocks, 1):
        progress = np.multiply(np.divide(iblock + 1.0, totalBlocks), 100.0)
        if (logFile == ""):
            sys.stdout.write("Reading... [%3.2f%%]\r" % progress)
            sys.stdout.flush()
        else:
            logFile.write("Reading... [%3.2f%%]\n" % progress)
        
        lobin = int(np.add(np.multiply(iblock, BLOCKSIZE), startbin))
        hibin = int(np.add(lobin, BLOCKSIZE))
        read_nspec = hibin-lobin
        spectra = fb.get_spectra(lobin, read_nspec)
        
        spectraData[:, lobin:hibin] = spectra[:, :]
        
        #for ichan in np.arange(0, totalChans, 1):
        #    print(ichan)
        #    spectraData[ichan, lobin:hibin] = spectra[ichan, :]
    
    if (remainder):
        progress = np.multiply(np.divide(iblock + 2.0, totalBlocks), 100.0)
       
        if (logFile == ""):
            sys.stdout.write("Reading... [%3.2f%%]\r" % progress)
            sys.stdout.flush()
        else:
            logFile.write("Reading... [%3.2f%%]\n" % progress)
        
        lobin = int(np.subtract(endbin, remainder))
        hibin = int(endbin)
        read_nspec = hibin-lobin
        spectra = fb.get_spectra(lobin, read_nspec)
       
        spectraData[:, lobin:hibin] = spectra[:, :] 
        #for ichan in np.arange(0, totalChans, 1):
        #    spectraData[ichan, lobin:hibin] = spectra[ichan, :]
    
    if (logFile == ""):
        print("\n")
    else:
        logFile.write("\n")
    
    return spectraData, inputHeader, inputNbits, h5pyFile;



def writeFilterbank(outputFilename, spectraData, inputHeader, inputNbits, 
                    logFile="", BLOCKSIZE = 1e6):
    """ 
    Write the filterbank data from memory to a filterbank file. 
    """
    if (logFile == ""):
        print("Writing filterbank file (%s)...\n" % outputFilename)
    else:
        logFile.write("Writing filterbank file (%s)...\n\n" % outputFilename)
    
    filterbank.create_filterbank_file(outputFilename, inputHeader, nbits=inputNbits)
    outfil = filterbank.FilterbankFile(outputFilename, mode='write')
    
    startbin = 0
    endbin = np.shape(spectraData)[1]
    
    nblocks = int(np.divide(endbin, BLOCKSIZE))
    remainder = endbin % BLOCKSIZE
    totalBlocks = nblocks
    
    if (remainder):
        totalBlocks = nblocks + 1
    
    for iblock in np.arange(0, nblocks, 1):
        progress = np.multiply(np.divide(iblock + 1.0, totalBlocks), 100.0)
        if (logFile == ""):
            sys.stdout.write("Writing... [%3.2f%%]\r" % progress)
            sys.stdout.flush()
        else:
            logFile.write("Writing... [%3.2f%%]\n" % progress)
        
        lobin = int(np.add(np.multiply(iblock, BLOCKSIZE), startbin))
        hibin = int(np.add(lobin, BLOCKSIZE))
        
        spectra = spectraData[:,lobin:hibin].T
        outfil.append_spectra(spectra)

    if (remainder):
        progress = np.multiply(np.divide(iblock + 2.0, totalBlocks), 100.0)
        if (logFile == ""):
            sys.stdout.write("Writing... [%3.2f%%]\r" % progress)
            sys.stdout.flush()
        else:
            logFile.write("Writing... [%3.2f%%]\n" % progress)
        
        lobin = int(np.subtract(endbin, remainder))
        hibin = int(endbin)
        
        spectra = spectraData[:,lobin:hibin].T
        outfil.append_spectra(spectra)
    
    if (logFile == ""):
        print("\n")
    else:
        logFile.write("\n")
    
    return;

