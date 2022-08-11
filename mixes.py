
import time
import numpy as np
import scipy.integrate
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline
from plotly.subplots import make_subplots
import plotly.express as px

def timeseries_spectra(signals, simLength, transient, regionLabels, mode="html", folder="figures",
                       freqRange=[2,40], opacity=1, title=None, auto_open=True):

    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], column_widths=[0.7, 0.3], horizontal_spacing=0.15)

    timepoints = np.arange(start=transient, stop=simLength, step=len(signals[0])/(simLength-transient))

    cmap = px.colors.qualitative.Plotly

    freqs = np.arange(len(signals[0]) / 2)
    freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

    for i, signal in enumerate(signals):

        # Timeseries
        if len(signal) < 8000:
            fig.add_trace(go.Scatter(x=timepoints, y=signal, name=regionLabels[i], opacity=opacity,
                                     legendgroup=regionLabels[i], marker_color=cmap[i%len(cmap)]), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=timepoints[:8000], y=signal[:8000], name=regionLabels[i], opacity=opacity,
                                     legendgroup=regionLabels[i], marker_color=cmap[i%len(cmap)]), row=1, col=1)

        # Spectra
        fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
        fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT

        fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies


        fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                 marker_color=cmap[i%len(cmap)], name=regionLabels[i], opacity=opacity,
                                 legendgroup=regionLabels[i], showlegend=False), row=1, col=2)


        fig.update_layout(xaxis=dict(title="Time (ms)"), xaxis2=dict(title="Frequency (Hz)"),
                          yaxis=dict(title="Voltage (mV)"), yaxis2=dict(title="Power (dB)"),
                          template="plotly_white", title=title, height=400, width=900)

    if title is None:
        title = "TEST_noTitle"

    if mode == "html":
        pio.write_html(fig, file=folder + "/" + title + ".html", auto_open=auto_open)
    elif mode == "png":
        pio.write_image(fig, file=folder + "/TimeSeries_" + str(time.time()) + ".png", engine="kaleido")
    elif mode == "svg":
        pio.write_image(fig, file=folder + "/" + title + ".svg", engine="kaleido")

    elif mode == "inline":
        plotly.offline.iplot(fig)





