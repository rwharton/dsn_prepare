import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def read_bpass(bp_file, ftype=0):
    """
    Read in a bandpass file and return frequencies 
    and bandpass coefficients 
    
    ftype = 0: *.bpass from sigproc bandpass
    ftype = 1: *.bandpass from prepfil
    """
    freqs = []
    bpvals = []
    
    if ftype == 0:
        ncols = 2
        fnum = 0
        bnum = 1
    elif ftype == 1:
        ncols = 3 
        fnum = 1
        bnum = 2
    else:
        print("ftype must be 0 or 1")
        return 0

    with open(bp_file, 'r') as fin:
        for line in fin:
            if line[0] == "#":
                continue
            else: pass 
            cols = line.split()
            if len(cols) != ncols:
                continue
            else: pass

            freq = float(cols[fnum])
            bp   = float(cols[bnum])
    
            freqs.append(freq)
            bpvals.append(bp)

    freqs = np.array(freqs)
    bpvals = np.array(bpvals)

    return freqs, bpvals


def bpass_from_subs(sub_bfiles, ftype=0):
    """
    Read in data from a list of subband files 
    and produce a combined bandpass
    """
    freqs = []
    bvals = []

    for sub in sub_bfiles:
        fs, bs = read_bpass(sub, ftype=ftype)
        freqs.append(fs)
        bvals.append(bs)

    freqs = np.hstack(freqs)
    bvals = np.hstack(bvals)

    xx = np.argsort(freqs)

    freqs = freqs[xx]
    bvals = bvals[xx]
    
    return freqs, bvals


def plot_bandpass(freqs, bpass, outfile, sub_chan=-1, 
                  title=None):
    """
    Make a plot of the bandpass and save it 
    to [outfile].png

    sub_chan is number of channels in one sub-band 
    only used for coloring purposes (if desired)
    """ 
    plt.ioff()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    Nchan = len(freqs)
    if sub_chan <= 0:
        ax.plot(freqs, bpass, ls='', marker='o')
    else:
        for ii in range( int(Nchan / sub_chan) ):
            xx = slice(ii * sub_chan, (ii+1) * sub_chan, 1)
            ax.plot(freqs[xx], bpass[xx], marker='o', ls='') 
     
    ax.set_yscale('log')

    if title is not None:
        ax.set_title(title, fontsize=18)
    
    ax.set_xlabel("Frequency (MHz)", fontsize=16)
    ax.set_ylabel("Bandpass Value", fontsize=16)

    plt.savefig("%s.png" %outfile, dpi=150, bbox_inches='tight')
    plt.close()
    plt.ion()

    return


def make_bpass_plots(indir, outdir):
    """
    Make plots for all the bandpasses 
    """ 
    t0_files = glob('%s/*bpass' %indir)
    t1_files = glob('%s/*bandpass' %indir)

    # Make t0 plot
    if len(t0_files):
        freqs0, bpass0 = bpass_from_subs(t0_files, ftype=0)
        sub_chans = int( len(freqs0) / len(t0_files) )

        t0_fn0 = t0_files[0].split('/')[-1]
        t0_base = t0_fn0.split('_')[0]
        t0_out = '%s/%s_initial_bp' %(outdir, t0_base)

        plot_bandpass(freqs0, bpass0, t0_out, sub_chan=sub_chans, 
                      title='Initial Bandpass')

    # Make t1 plots 
    for t1_file in t1_files:
        print(t1_file)
        freqs1, bpass1 = read_bpass(t1_file, ftype=1)
        t1_fn1 = t1_file.split('/')[-1]
        t1_base = t1_fn1.split('.bandpass')[0]
        t1_out = '%s/%s' %(outdir, t1_base)
        sub_chans1 = int(len(freqs1) / len(t0_files))
        plot_bandpass(freqs1, bpass1, t1_out, sub_chan=sub_chans1,   
                      title=t1_base)

    return      



 
