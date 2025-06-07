import time
import numpy as np
import scipy.integrate
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline

from pyToolbox.signals import epochingTool


## Postdoc functions - from 03/2025 @Jescab01
def FFT(signals, simLength, freq_range=[0.5, 60], norm=False):
    """

    :param signals: channels x timepoints
    :param simLength: in seconds
    :param norm: do you want normalized spectra?
    :param freq_range: Exclude some frequencies from analysis?
    :return:
    """

    if signals.ndim == 1:

        fft = abs(np.fft.fft(signals))  # FFT for each channel signal
        fft = fft[range(int(len(signals) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals) / 2)
        freqs = freqs / simLength  # simLength (s)

        # Filter out undesired frequencies
        fft = fft[(freqs > freq_range[0]) & (freqs < freq_range[1])]
        freqs = freqs[(freqs > freq_range[0]) & (freqs < freq_range[1])]

        if norm:
            # fft = fft / sum(fft)
            fft_norm = (fft - min(fft)) / (max(fft) - min(fft))
            return fft, freq, fft_norm
        else:
            return np.asarray(fft), freqs

    elif signals.ndim == 2:
        ffts, ffts_norm = [], []

        for i in range(len(signals)):
            fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
            fft = fft[range(int(len(signals[i]) / 2))]  # Select just positive side of the symmetric FFT
            freqs = np.arange(len(signals[i]) / 2)
            freqs = freqs / simLength  # simLength (s)

            fft = fft[(freqs > freq_range[0]) & (freqs < freq_range[1])]  # remove undesired frequencies from peak analysis
            freqs = freqs[(freqs > freq_range[0]) & (freqs < freq_range[1])]

            ffts.append(fft)

            if norm:
                # fft = fft / sum(fft)
                fft = (fft - min(fft))/(max(fft) - min(fft))

            ffts_norm.append(fft)

        if norm:
            return np.asarray(ffts), freqs, np.asarray(ffts_norm)
        else:
            return np.asarray(ffts), freqs

    else:
        print("Array should be an array with shape: channels x timepoints.")


def FFTpeaks(signals, simLength, freq_range=[0.5, 60]):
    """

    :param signals: channels x timepoints
    :param simLength: in seconds
    :param norm: do you want normalized spectra?
    :param freq_range: Exclude some frequencies from analysis?
    :return:
    """

    if signals.ndim == 1:

        fft = abs(np.fft.fft(signals))  # FFT for each channel signal
        fft = fft[range(int(len(signals) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals) / 2)
        freqs = freqs / simLength  # simLength (s)

        # Filter out undesired frequencies
        fft = fft[(freqs > freq_range[0]) & (freqs < freq_range[1])]
        freqs = freqs[(freqs > freq_range[0]) & (freqs < freq_range[1])]

        # Guided by max fft
        peak = freqs[np.where(fft == max(fft))][0]
        module = fft[np.where(fft == max(fft))][0]
        band_module = scipy.integrate.simpson(fft[(peak - 2 < freqs) & (freqs < peak + 2)])  # integral

        return peak, module, band_module

    elif signals.ndim == 2:

        peaks, modules, band_modules = [], [], []

        for i in range(len(signals)):
            fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
            fft = fft[range(int(len(signals[i]) / 2))]  # Select just positive side of the symmetric FFT
            freqs = np.arange(len(signals[i]) / 2)
            freqs = freqs / simLength  # simLength (s)

            fft = fft[(freqs > freq_range[0]) & (freqs < freq_range[1])]  # remove undesired frequencies from peak analysis
            freqs = freqs[(freqs > freq_range[0]) & (freqs < freq_range[1])]

            # Guided by max fft
            peak = freqs[np.where(fft == max(fft))][0]
            peaks.append(peak)
            modules.append(fft[np.where(fft == max(fft))][0])
            band_modules = scipy.integrate.simpson(fft[(peak - 2 < freqs) & (freqs < peak + 2)])  # integral

        return peaks, modules, band_modules

    else:
        print("Array should be an array with shape: channels x timepoints.")




# Predoc functions - old.
def multitapper(signals, samplingFreq, regionLabels, epoch_length=4, ntapper=4, smoothing=0.5, peaks=False,
                plot=False, folder="figures/", title="", mode="html"):

    # Demean data
    demean_data = signals - np.average(signals, axis=1)[:, None]

    # Epoch data
    nsamples = epoch_length * 1000
    epoched_data = epochingTool(demean_data, epoch_length, 1000, verbose=False)

    # Generate the slepian tappers
    dpss = scipy.signal.windows.dpss(nsamples, nsamples * (smoothing / samplingFreq), ntapper)

    # select frequency band
    freqs = np.arange(nsamples - 1) / (nsamples / samplingFreq)
    limits = [2, 40]
    freqsindex = np.where((freqs >= limits[0]) & (freqs <= limits[1]))

    # Applies the tapper
    multitappered_data = np.asarray([filter * epoched_data for filter in dpss])

    # Fourier transform of the filtered data
    fft_multitappered_data = np.fft.fft(multitappered_data).real / np.sqrt(nsamples)

    # Power calculation
    power_data = abs(fft_multitappered_data) ** 2

    # Average FFTs by trials and tappers
    avg_power_data = np.average(power_data, (0, 1))

    # Keep selected frequencies
    ffreqs = freqs[freqsindex]
    fpower = np.squeeze(avg_power_data[:, freqsindex])

    if plot == True:
        fig = go.Figure(
            layout=dict(title="Brainstorm processed data - ROIs spectra (Py)", xaxis=dict(title='Frequency (Hz)'),
                        yaxis=dict(title='Power')))
        for roi in range(len(signals)):
            fig.add_trace(go.Scatter(x=ffreqs, y=fpower[roi], mode="lines", name=regionLabels[roi]))

        if mode == "html":
            pio.write_html(fig, file=folder + "BS_" + title + "_Spectra.html", auto_open=True)
        elif mode == "inline":
            plotly.offline.iplot(fig)


    if peaks:
        IAF = ffreqs[fpower.argmax(axis=1)]
        modules = [fpower[spectrum, fpower.argmax(axis=1)[spectrum]] for spectrum in range(len(fpower))]
        band_modules = [scipy.integrate.simpson(fpower[spectrum, (IAF[spectrum] - 2 < ffreqs) & (ffreqs < IAF[spectrum] + 2)]) for spectrum in range(len(fpower))]  # Alpha band integral
        return ffreqs, fpower, IAF, np.asarray(modules), np.asarray(band_modules)

    else:
        return ffreqs, fpower


def FFTarray(signals, simLength, transient, regionLabels, param1=None, param2=None, param3=None, lowcut=5, highcut=15):

    regLabs = list()
    fft_tot = list()
    freq_tot = list()
    param1array = list()
    param2array = list()
    param3array = list()

    for i in range(len(signals)):
        fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
        fft = fft[range(int(len(signals[i]) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals[i]) / 2)
        freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

        fft = fft[(freqs > lowcut) & (freqs < highcut)]  # remove undesired frequencies
        freqs = freqs[(freqs > lowcut) & (freqs < highcut)]

        regLabs += [regionLabels[i]] * len(fft)
        fft_tot += list(fft)
        freq_tot += list(freqs)
        param1array += [param1] * len(fft)
        param2array += [param2] * len(fft)
        param3array += [param3[i, 0]] * len(fft)

    return np.asarray([param1array, param2array, regLabs, fft_tot, freq_tot, param3array], dtype=object).transpose()


def PSD(signals, samplingFreq, window=4, overlap=0.5):
    """

    :param signals: nchannels x ntimepoints
    :param samplingFreq:
    :param window: def=4secs
    :param overlap: def=0.5
    :return:
    """

    fft_result = []

    window_size = int(window * samplingFreq)
    step_size = int(window_size * (1 - overlap))

    for roi, roi_signal in enumerate(signals):
        eSignal = [roi_signal[i: i + window_size] for i in range(0, len(roi_signal) - window_size, step_size)]
        fft_vector = []
        for epoch in eSignal:
            fft_temp = abs(np.fft.fft(epoch))  # FFT for each channel signal
            fft_temp = fft_temp[range(int(len(epoch) / 2))]  # Select just positive side of the symmetric FFT
            fft_vector.append(fft_temp)

        fft_result.append(np.average(fft_vector, axis=0))
    freqs = np.arange(window_size / 2)
    freqs = freqs / window

    return np.asarray(fft_result), freqs


def PSDplot(signals, samplingFreq, regionLabels, folder="figures", title=None, mode="html", highcut=80, lowcut=1,
            type="log", window=4, overlap=0.5):

    """

    :param signals:
    :param samplingFreq:
    :param regionLabels:
    :param folder:
    :param title:
    :param mode:
    :param max_hz:
    :param min_hz:
    :param type:
    :param window:
    :param overlap:
    :return:
    """

    fig = go.Figure(layout=dict(title=title, xaxis=dict(title='Frequency', type=type), yaxis=dict(title='Log power (dB)', type=type)))

    window_size = int(window * samplingFreq)
    step_size = int(window_size * (1 - overlap))

    for roi, roi_signal in enumerate(signals):
        eSignal = [roi_signal[i: i + window_size] for i in range(0, len(roi_signal) - window_size, step_size)]
        fft_vector = []
        for epoch in eSignal:
            fft_temp = abs(np.fft.fft(epoch))  # FFT for each channel signal
            fft_temp = fft_temp[range(int(len(epoch) / 2))]  # Select just positive side of the symmetric FFT
            fft_vector.append(fft_temp)

        fft = np.average(fft_vector, axis=0)
        freqs = np.arange(window_size / 2)
        freqs = freqs / window

        cut_high = np.where(freqs >= highcut)[0][0]  # Hz. Number of frequency points until cut at xHz point
        cut_low = np.where(freqs >= lowcut)[0][0]
        fig.add_scatter(x=freqs[int(cut_low):int(cut_high)], y=fft[int(cut_low):int(cut_high)], name=regionLabels[roi])

    if mode == "html":
        pio.write_html(fig, file=folder + "/FFT_" + title + ".html", auto_open=True)
    elif mode == "png":
        pio.write_image(fig, file=folder + "/FFT_" + str(time.time()) + ".png", engine="kaleido")
    elif mode == "inline":
        plotly.offline.iplot(fig)


def FFTplot(signals, simLength, regionLabels, folder="figures", title=None, mode="html", max_hz=80, min_hz=1,
            type="log", auto_open=True):

    fig = go.Figure(layout=dict(title=title, xaxis=dict(title='Frequency', type=type), yaxis=dict(title='Module', type=type)))

    for i in range(len(signals)):
        fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
        fft = fft[range(int(len(signals[0]) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals[0]) / 2)
        freqs = freqs / (simLength / 1000)  # simLength (ms) / 1000 -> segs

        cut_high = (simLength / 2) / 500 * max_hz  # Hz. Number of frequency points until cut at xHz point
        cut_low = (simLength / 2) / 500 * min_hz
        fig.add_scatter(x=freqs[int(cut_low):int(cut_high)], y=fft[int(cut_low):int(cut_high)], name=regionLabels[i])

    if mode == "html":
        pio.write_html(fig, file=folder + "/FFT_" + title + ".html", auto_open=auto_open)
    elif mode == "png":
        pio.write_image(fig, file=folder + "/FFT_" + str(time.time()) + ".png", engine="kaleido")
    elif mode == "inline":
        plotly.offline.iplot(fig)


