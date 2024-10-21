#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:40:25 2023

@author: bru
"""


import re
import datetime

import mne
import numpy
import scipy.signal
from scipy import fft


# Lists the valid MNE objects.
mnevalid = (
    mne.io.BaseRaw,
    mne.BaseEpochs )

# Sets the verbosity level for MNE.
mne.set_log_level ( verbose = 'ERROR' )



# Function for two-pass filtering on MNE objects.
def filtfilt ( mnedata, num = 1, den = 1, hilbert = False ):
    """ Wrapper to apply two-pass filtering to MNE objects."""
    
    
    # Checks if the data is a valid MNE object.
    if not isinstance ( mnedata, mnevalid ):
        
        print ( 'Unsupported data type.' )
        return None
    
    
    # Creates a copy of the input data.
    mnedata  = mnedata.copy ()

    # Gets the raw data matrix.
    rawdata  = mnedata.get_data ()


    # For IIR filters uses SciPy (faster and more accurate).
    if numpy.array ( den ).size != 1:

        # Gets the data metadata.
        dshape   = rawdata.shape
        nsample  = dshape [-1]

        # Reshapes the data into a 2D array.
        rawdata  = rawdata.reshape ( ( -1, nsample ) )

        # Filters the data.
        rawdata  = scipy.signal.filtfilt (
            num,
            den,
            rawdata )

        # Restores the original data shape.
        rawdata  = rawdata.reshape ( dshape )

    # For FIR filters use FFT (much faster, same accuracy).
    else:

        # Filters the data.
        rawdata  = signal_filtfilt (
            rawdata,
            num = num,
            den = den,
            hilbert = hilbert )
    
    # Replaces the data and marks it as loaded.
    mnedata._data = rawdata
    mnedata.preload = True
    
    ## Creates a new MNE object with the filtered data.
    #mnedata   = mne.EpochsArray ( rawdata, data.info, events = data.events, verbose = False )
    
    ## Creates a new MNE object with the filtered data.
    #mnedata    = mne.io.RawArray ( rawdata, data.info, verbose = False )
    
    # Returns the MNE object.
    return mnedata



def decimate (
        mnedata,
        ratio = 1 ):
    """Decimates an MNE object with no filtering."""
    """
    Based on MNE 1.7 functions:
    mne.BaseRaw.resample
    https://github.com/mne-tools/mne-python/blob/maint/1.7/mne/io/base.py
    mne.Epochs.decimate
    https://github.com/mne-tools/mne-python/blob/maint/1.7/mne/utils/mixin.py
    """


    # Checks if the data is a valid MNE object.
    if not isinstance ( mnedata, mnevalid ):

        print ( 'Unsupported data type.' )
        return None


    # Ratio is 1 does nothing.
    if ratio == 1:
        return mnedata


    # Creates a copy of the input data.
    mnedata  = mnedata.copy ()


    # Gets the raw data matrix.
    rawdata  = mnedata.get_data ()

    # Decimates the raw data matrix in the last dimension.
    decdata = rawdata [ ..., :: ratio ]

    # Replaces the data and marks it as loaded.
    mnedata._data = decdata
    mnedata.preload = True


    # Updates the sampling rate.
    with mnedata.info._unlock ():
        mnedata.info [ 'sfreq' ] = mnedata.info [ 'sfreq' ] / ratio


    # Updates the mne.Raw information.
    if isinstance ( mnedata, mne.io.BaseRaw ):
        n_news = numpy.array(decdata.shape[1:])
        mnedata._cropped_samp = int ( numpy.round ( mnedata._cropped_samp * ratio ) )
        mnedata._first_samps = numpy.round ( mnedata._first_samps * ratio ).astype ( int )
        mnedata._last_samps = numpy.array ( mnedata._first_samps ) + n_news - 1
        mnedata._raw_lengths [ :1 ] = list ( n_news )

    # Updates the mne.Epochs information.
    if isinstance ( mnedata, mne.BaseEpochs ):
        mnedata._decim = 1
        mnedata._set_times ( mnedata._raw_times [ :: ratio ] )
        mnedata._update_first_last ()


    # Returns the MNE object.
    return mnedata


# Function for two-pass filtering.
def signal_filtfilt(data, num=1, den=1, hilbert=False):
    ''' Filters the provided data in two passes.'''

    # Sanitizes the inputs.
    data = numpy.array(data)
    num = numpy.array(num)
    den = numpy.array(den)

    if data.ndim == 0:
        data = data.reshape(-1)
    if num.ndim == 0:
        num = num.reshape(-1)
    if den.ndim == 0:
        den = den.reshape(-1)

    # Gets the data metadata.
    dshape = data.shape
    nsample = dshape[-1]
    real = numpy.isreal(data).all()

    # Reshapes the data into a 2D array.
    data = data.reshape((-1, nsample))

    # Gets the filter metadata.
    norder = num[:-1, ].size
    dorder = den[:-1, ].size
    order = norder + dorder

    # Estimates the optimal chunk and FFT sizes.
    nfft = optnfft(nsample, order)
    chsize = nfft - 2 * order

    # Calculates the butterfly reflections of the data.
    prepad = 2 * data[:, :1] - data[:, order: 0: -1]
    pospad = 2 * data[:, -1:] - data[:, -2: -order - 2: -1]

    # Adds the reflections as padding.
    paddata = numpy.concatenate((prepad, data, pospad), axis=-1)

    # Gets the Fourier transform of the filter.
    Fnum = fft.fft(num, n=nfft, axis=-1, norm=None)
    Fden = fft.fft(den, n=nfft, axis=-1, norm=None)

    # Combines all the numerators and denominators.
    if Fnum.ndim > 1:
        Fnum = Fnum.prod(axis=0, keepdims=True)
    if Fden.ndim > 1:
        Fden = Fden.prod(axis=0, keepdims=True)

    # Combines numerator and denominator.
    Ffilter = Fnum / Fden

    # Gets the squared absolute value of the filter (two-passes).
    Ffilter = Ffilter * Ffilter.conjugate()

    # Applies Hilbert transform, if required.
    if hilbert:
        # Lists the positive and negative part of the spectra.
        spos = (fft.fftfreq(nfft) > 0) & (fft.fftfreq(nfft) < 0.5)
        sneg = (fft.fftfreq(nfft) < 0) & (fft.fftfreq(nfft) > -0.5)

        # Removes the negative part of the filter spectrum.
        Ffilter[sneg,] = 0

        # Duplicates the positive part of the filter spectrum.
        Ffilter[spos,] = Ffilter[spos,] * 2

        # Converts the input data into complex.
        data = data + 0j

    # Goes through each data chunk.
    for index in range(0, numpy.ceil(nsample / chsize).astype(int)):

        # Calculates the offset and length for the current chunk.
        offset = index * chsize
        chlen = numpy.min((chsize, nsample - offset))

        # Gets the chunk plus the padding.
        chunk = paddata[:, offset: offset + chlen + 2 * order].copy()

        # Takes the Fourier transform of the chunk.
        Fchunk = fft.fft(chunk, n=nfft, axis=-1, norm=None, workers=1)  # Orig -1

        # Applies the filter.
        Fchunk = Fchunk * Ffilter

        # Recovers the filtered chunk.
        chunk = fft.ifft(Fchunk, n=nfft, axis=-1, workers=1)

        # Gets only the real part, if required.
        if real and not hilbert:
            chunk = chunk.real

        # Stores the filtered chunk of data.
        data[:, offset: offset + chlen] = chunk[:, order: order + chlen]

    # Restores the data shape.
    data = data.reshape(dshape)

    # Returns the filtered data.
    return data


# Function to get the optimal chunk size for the FFT.
def optnfft(nsample, order=0):
    '''Returns the optimal length of the FFT.'''

    # Uses the closest multiple of 512 * 5 samples larger than 10 filters.
    nfft = 512 * 5 * numpy.ceil((10 * order) / (512 * 5))
    nfft = nfft.astype(int)

    # If lower than 51200, uses 51200.
    nfft = numpy.max((nfft, 51200))

    # For checking with Matlab.
    nfft = numpy.min((50000, nsample)) + 2 * order

    # Returns the optimal chunck size.
    return nfft