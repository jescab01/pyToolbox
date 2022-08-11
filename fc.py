import time
from collections import Counter

import numpy as np
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.offline

def CORR(signals, regionLabels, plot="OFF"):
    """
    To compute correlation between signals you need to standarize signal values and then to sum up intersignal products
    divided by signal length.
    """

    normalSignals = np.ndarray((len(signals), len(signals[0])))
    for channel in range(len(signals)):
        mean = np.mean(signals[channel, :])
        std = np.std(signals[channel, :])
        normalSignals[channel] = (signals[channel, :] - mean) / std

    CORR = np.ndarray((len(normalSignals), len(normalSignals)))
    for channel1 in range(len(normalSignals)):
        for channel2 in range(len(normalSignals)):
            CORR[channel1][channel2] = sum(normalSignals[channel1] * normalSignals[channel2]) / len(normalSignals[0])

    if plot == "ON":
        fig = go.Figure(data=go.Heatmap(z=CORR, x=regionLabels, y=regionLabels, colorscale='Viridis'))
        fig.update_layout(title='Correlation')
        pio.write_html(fig, file="figures/CORR.html", auto_open=True)

    return CORR


def PLV(efPhase, regionLabels=None, folder=None, plot=None, verbose=True, auto_open=False):
    tic = time.time()
    try:
        efPhase[0][0][0]  # Test whether signals have been epoched
        PLV = np.ndarray((len(efPhase[0]), len(efPhase[0])))

        if verbose:
            print("Calculating PLV", end="")
        for channel1 in range(len(efPhase[0])):
            if verbose:
                print(".", end="")
            for channel2 in range(len(efPhase[0])):
                plv_values = list()
                for epoch in range(len(efPhase)):
                    phaseDifference = efPhase[epoch][channel1] - efPhase[epoch][channel2]
                    value = abs(np.average(np.exp(1j * phaseDifference)))
                    plv_values.append(value)
                PLV[channel1, channel2] = np.average(plv_values)

        if plot:
            fig = go.Figure(data=go.Heatmap(z=PLV, x=regionLabels, y=regionLabels,
                                            colorbar=dict(thickness=4), colorscale='Viridis'))
            fig.update_layout(title='Phase Locking Value', height=500, width=500)

            if plot == "html":
                pio.write_html(fig, file=folder + "/PLV.html", auto_open=auto_open)
            elif plot == "png":
                pio.write_image(fig, file=folder + "/PLV_" + str(time.time()) + ".png", engine="kaleido")
            elif plot == "svg":
                pio.write_image(fig, file=folder + "/PLV.svg", engine="kaleido")
            elif plot == "inline":
                plotly.offline.iplot(fig)

        if verbose:
            print("%0.3f seconds.\n" % (time.time() - tic,))
        return PLV

    except IndexError:
        print("IndexError. Signals must be epoched. Use epochingTool().")


def PLI(efPhase, regionLabels=None, folder=None, subject="", plot="OFF", auto_open=False):
    tic = time.time()
    try:
        efPhase[0][0][0]
        PLI = np.ndarray(((len(efPhase[0])), len(efPhase[0])))

        print("Calculating PLI", end="")
        for channel1 in range(len(efPhase[0])):
            print(".", end="")
            for channel2 in range(len(efPhase[0])):
                pli_values = list()
                for epoch in range(len(efPhase)):
                    phaseDifference = efPhase[epoch][channel1] - efPhase[epoch][channel2]
                    value = np.abs(np.average(np.sign(np.sin(phaseDifference))))
                    pli_values.append(value)
                PLI[channel1, channel2] = np.average(pli_values)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=PLI, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Phase Lag Index')
            pio.write_html(fig, file=folder + "/PLI_" + subject + ".html", auto_open=auto_open)

        print("%0.3f seconds.\n" % (time.time() - tic,))
        return PLI

    except IndexError:
        print("IndexError. Signals must be epoched. Use epochingTool().")


def AEC(efEnvelope, regionLabels=None, folder=None, subject="", plot="OFF", auto_open=False):
    tic = time.time()
    try:
        efEnvelope[0][0][0]
        AEC = np.ndarray(((len(efEnvelope[0])), len(efEnvelope[0])))  # averaged AECs per channel x channel

        print("Calculating AEC", end="")
        for channel1 in range(len(efEnvelope[0])):
            print(".", end="")
            for channel2 in range(len(efEnvelope[0])):
                values_aec = list()  # AEC per epoch and channel x channel
                for epoch in range(len(efEnvelope)):  # CORR between channels by epoch
                    r = np.corrcoef(efEnvelope[epoch][channel1], efEnvelope[epoch][channel2])
                    values_aec.append(r[0, 1])
                AEC[channel1, channel2] = np.average(values_aec)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=AEC, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Amplitude Envelope Correlation')
            pio.write_html(fig, file=folder + "/AEC_" + subject + ".html", auto_open=auto_open)

        print("%0.3f seconds.\n" % (time.time() - tic,))
        return AEC

    except IndexError:
        print("IndexError. Signals must be epoched before calculating AEC. Use epochingTool().")
