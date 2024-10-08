import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline
from plotly.subplots import make_subplots


# Signals
def timeseriesPlot(signals, timepoints, regionLabels, folder="figures", title=None, mode="html", auto_open=True):
    fig = go.Figure(layout=dict(title=title, xaxis=dict(title='time (ms)'), yaxis=dict(title='Voltage')))
    for ch in range(len(signals)):
        fig.add_scatter(x=timepoints, y=signals[ch], name=regionLabels[ch])

    if title is None:
        title = "TimeSeries"

    if mode == "html":
        pio.write_html(fig, file=folder + "/" + title + ".html", auto_open=auto_open)
    elif mode == "png":
        pio.write_image(fig, file=folder + "/TimeSeries_" + str(time.time()) + ".png", engine="kaleido")
    elif mode == "inline":
        plotly.offline.iplot(fig)


def epochingTool(data, window, step, samplingFreq, msg="", verbose=True):
    """
    Epoch length in seconds; sampling frequency in Hz
    """

    tic = time.time()

    if verbose:
        print("Epoching %s" % msg, end="")

    window_, step_ = int(window*samplingFreq), int(step*samplingFreq)

    epochedSignals = [data[:, w:w + window_] for w in np.arange(0, data.shape[-1] - window_, step_, 'int')]

    if verbose:
        print(" - %0.2e seconds.\n" % (time.time() - tic,))

    return epochedSignals


def bandpassFIRfilter(signals, lowcut, highcut, windowtype, samplingRate, times=None, plot="OFF"):
    """
     Truncated sinc function in whatever order, needs to be windowed to enhance frequency response at side lobe and rolloff.
     Some famous windows are: bartlett, hann, hamming and blackmann.
     Two processes depending on input: epoched signals or entire signals
     http://www.labbookpages.co.uk/audio/firWindowing.html
     https://en.wikipedia.org/wiki/Window_function
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    """
    tic = time.time()
    try:
        signals[0][0][0]
        order = int(len(signals[0][0]) / 3)
        firCoeffs = firwin(numtaps=order + 1, cutoff=[lowcut, highcut], window=windowtype, pass_zero="bandpass",
                           fs=samplingRate)
        efsignals = np.ndarray((len(signals), len(signals[0]), len(signals[0][0])))
        print("Band pass filtering epoched signals: %i-%iHz " % (lowcut, highcut), end="")
        for channel in range(len(signals)):
            print(".", end="")
            for epoch in range(len(signals[0])):
                efsignals = np.ndarray((len(signals), len(signals[channel]), len(signals[channel][epoch])))
                efsignals[channel][epoch] = filtfilt(b=firCoeffs, a=[1.0], x=signals[channel][epoch],
                                                     padlen=int(2.5 * order))
                # a=[1.0] as it is FIR filter (not IIR).
        print("%0.3f seconds.\n" % (time.time() - tic,))
        return efsignals

    except IndexError:
        order = int(len(signals[0, :]) / 3)
        firCoeffs = firwin(numtaps=order + 1, cutoff=[lowcut, highcut], window=windowtype, pass_zero="bandpass",
                           fs=samplingRate)
        filterSignals = filtfilt(b=firCoeffs, a=[1.0], x=signals,
                                 padlen=int(2.5 * order))  # a=[1.0] as it is FIR filter (not IIR).
        if plot == "ON":
            plt.plot(range(len(firCoeffs)), firCoeffs)  # Plot filter shape
            plt.title("FIR filter shape w/ %s windowing" % windowtype)
            for i in range(1, 10):
                plt.figure(i + 1)
                plt.xlabel("time (ms)")
                plt.plot(times, signals[i], label="Raw signal")
                plt.plot(times, filterSignals[i], label="Filtered Signal")
            plt.show()
            plt.savefig("figures/filterSample%s" % str(i))
        return filterSignals


def plotConversions(raw_signals, filterSignals, phase, amplitude_envelope, band, regionLabels=None, n_signals=1,
                    raw_time=None):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for channel in range(n_signals):

        fig.add_scatter(x=raw_time, y=raw_signals[channel], name="Raw signal")
        fig.add_scatter(x=raw_time, y=filterSignals[channel], name="Filtered signal")
        fig.add_scatter(x=raw_time, y=phase[channel], name="Instantaneous phase", secondary_y=True)
        fig.add_scatter(x=raw_time, y=amplitude_envelope[channel], name="Amplitude envelope")

        fig.update_layout(title="%s filtered - %s signal conversions" % (band, regionLabels[channel]))
        fig.update_xaxes(title_text="Time (ms)")

        fig.update_yaxes(title_text="Amplitude", range=[-max(raw_signals[channel]), max(raw_signals[channel])],
                         secondary_y=False)
        fig.update_yaxes(title_text="Phase", tickvals=[-3.14, 0, 3.14], range=[-15, 15], secondary_y=True,
                         gridcolor='mintcream')

        pio.write_html(fig, file="figures/%s_%s_conversions.html" % (band, regionLabels[channel]), auto_open=True)
