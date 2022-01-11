# This is going to be a script with useful functions I will be using frequently.
import math
import time
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.signal import firwin, filtfilt, hilbert
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.offline
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px


# ParamSpace y otras cosas
def others(df, z=None, title=None, names=None, folder="figures", auto_open="True", show_owp=False):

    if title == "FC_comparisons":
        bands = ["2-theta", "3-alfa", "4-beta", "5-gamma"]  # omiting first band in purpose
        fig = make_subplots(rows=2, cols=5, subplot_titles=(
        "Delta", "Theta", "Alpha", "Beta", "Gamma", "Delta", "Theta", "Alpha", "Beta", "Gamma"),
                            row_titles=("AEC", "PLV"), shared_yaxes=True, shared_xaxes=True)
        fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].r,
                                 x=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].frm,
                                 y=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].to,
                                 colorscale='Viridis', colorbar=dict(title="Pearson's r", thickness=10),
                                 zmin=min(df.r), zmax=1), row=1, col=1)
        fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].r,
                                 x=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].frm,
                                 y=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].to,
                                 colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=2, col=1)
        for i, b in enumerate(bands):
            fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "aec") & (df["band"] == b)].r,
                                     x=df[(df["FC_measure"] == "aec") & (df["band"] == b)].frm,
                                     y=df[(df["FC_measure"] == "aec") & (df["band"] == b)].to,
                                     colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=1, col=i + 2)
            fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == b)].r,
                                     x=df[(df["FC_measure"] == "plv") & (df["band"] == b)].frm,
                                     y=df[(df["FC_measure"] == "plv") & (df["band"] == b)].to,
                                     colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=2, col=i + 2)
        fig.update_layout(title_text="FC correlation between subjects")
        pio.write_html(fig, file=folder + "/%s.html" % title, auto_open=auto_open)

    elif title == "sf_comparisons":
        fig = make_subplots(rows=1, cols=2, subplot_titles=("PLV", "AEC"),
                            shared_yaxes=True, shared_xaxes=True,
                            x_title="Frequency bands")

        fig.add_trace(go.Heatmap(z=df[df["FC_measure"] == "aec"].r,
                                 x=df[df["FC_measure"] == "aec"].band,
                                 y=df[df["FC_measure"] == "aec"].subj,
                                 colorscale='Viridis', colorbar=dict(title="Pearson's r", thickness=10),
                                 zmin=min(df.r), zmax=max(df.r)), row=1, col=1)

        fig.add_trace(go.Heatmap(z=df[df["FC_measure"] == "plv"].r,
                                 x=df[df["FC_measure"] == "plv"].band,
                                 y=df[df["FC_measure"] == "plv"].subj,
                                 colorscale='Viridis', showscale=False,
                                 zmin=min(df.r), zmax=max(df.r)), row=1, col=2)
        fig.update_layout(
            title_text='Structural - Functional correlations')
        pio.write_html(fig, file=folder + "/%s.html" % title, auto_open=auto_open)

    elif title == "interW":
        fig = ff.create_annotated_heatmap(df, names, names, colorscale="Viridis", showscale=True,
                                          colorbar=dict(title="Pearson's r"))
        fig.update_layout(title_text="Correlations in sctructural connectivity between real subjects")
        pio.write_html(fig, file=folder + "/%s.html" % title, auto_open=auto_open)

    elif "fft-bm" in title:
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(z=df.mS_peak, x=df.speed, y=df.G, colorscale='Viridis', colorbar=dict(title="FFT peak (Hz)")))
        fig.add_trace(
            go.Scatter(text=np.round(df.mS_bm, 2), x=df.speed, y=df.G, mode="text", textfont=dict(color="crimson")))
        fig.update_layout(
            title_text="Heatmap for simulation's mean signal FFT peak; in red Hartigans' bimodality test's p-value. (0 -> p<2.2e-16)")
        fig.update_xaxes(title_text="Conduction speed (m/s)")
        fig.update_yaxes(title_text="Coupling factor")
        pio.write_html(fig, file=folder + "/paramSpace-%s.html" % title, auto_open=auto_open)

    else:
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(z=df.mS_peak, x=df.speed, y=df.G, colorscale='Viridis', colorbar=dict(title="FFT peak (Hz)")))

        fig.update_layout(
            title_text="FFT peak of simulated signals by Coupling factor and Conduction speed")
        fig.update_xaxes(title_text="Conduction speed (m/s)")
        fig.update_yaxes(title_text="Coupling factor")
        pio.write_html(fig, file=folder + "/paramSpace-FFTpeak_%s.html" % title, auto_open=auto_open)


def emp_sim():
    fig = make_subplots(rows=2, cols=5, subplot_titles=(
    "Delta", "Theta", "Alpha", "Beta", "Gamma", "Delta", "Theta", "Alpha", "Beta", "Gamma"),
                        row_titles=("AEC", "PLV"), shared_yaxes=True, shared_xaxes=True)
    fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].r,
                             x=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].frm,
                             y=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].to,
                             colorscale='Viridis', colorbar=dict(title="Pearson's r", thickness=10),
                             zmin=min(df.r), zmax=1), row=1, col=1)
    fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].r,
                             x=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].frm,
                             y=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].to,
                             colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=2, col=1)
    for i, b in enumerate(bands):
        fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "aec") & (df["band"] == b)].r,
                                 x=df[(df["FC_measure"] == "aec") & (df["band"] == b)].frm,
                                 y=df[(df["FC_measure"] == "aec") & (df["band"] == b)].to,
                                 colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=1, col=i + 2)
        fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == b)].r,
                                 x=df[(df["FC_measure"] == "plv") & (df["band"] == b)].frm,
                                 y=df[(df["FC_measure"] == "plv") & (df["band"] == b)].to,
                                 colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=2, col=i + 2)
    fig.update_layout(title_text="FC correlation between subjects")
    pio.write_html(fig, file=folder + "/%s.html" % title, auto_open=auto_open)

