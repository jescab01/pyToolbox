import numpy as np
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots


def fft(df, folder, title=None, auto_open="True"):
    fig = make_subplots(rows=2, cols=1, subplot_titles=(["FFT peak", "Average activity amplitude"]),
                        shared_yaxes=False, shared_xaxes=False)

    fig.add_trace(go.Heatmap(z=df.tFFT_peak, x=df.stimFreq, y=df.stimAmplitude,
                             colorscale='Viridis',
                             colorbar=dict(title="Hz", thickness=20, y=0.82, ypad=120),
                             zmin=np.min(df.tFFT_peak), zmax=np.max(df.tFFT_peak)), row=1, col=1)

    fig.add_trace(go.Heatmap(z=df.tAvg_activity, x=df.stimFreq, y=df.stimAmplitude,
                             colorscale='Inferno',
                             colorbar=dict(title="mV", thickness=20, y=0.2, ypad=120),
                             zmin=np.min(df.tAvg_activity), zmax=np.max(df.tAvg_activity)), row=2, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text="stimulation frequency", row=1, col=1)
    fig.update_xaxes(title_text="stimulation frequency", row=2, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="stimulus amplitude", row=1, col=1)
    fig.update_yaxes(title_text="stimulus amplitude", row=2, col=1)

    fig.update_layout(title_text=title)
    pio.write_html(fig, file=folder + "/" + title + ".html", auto_open=auto_open)


def fc(df, structure, regionLabels, folder, region="Left-Cuneus", title="w-conn", t_c="target",
                   auto_open="True"):
    if "FFT" not in region:

        sp_titles = ["Stimulation Weight = " + str(ws) for ws in set(df.stimAmplitude)]

        fig = make_subplots(rows=len(set(df.stimAmplitude)) + 1, cols=1,
                            subplot_titles=(sp_titles + ["Structural Connectivity"]),
                            shared_yaxes=True, shared_xaxes=True, y_title="Stimulation Frequency")

        for i, ws in enumerate(set(df.stimAmplitude)):
            subset = df[df["stimAmplitude"] == ws]

            if i == 0:
                fig.add_trace(
                    go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                               colorbar=dict(title="PLV"), zmin=0, zmax=1), row=i + 1, col=1)

            else:
                fig.add_trace(
                    go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                               zmin=0, zmax=1, showscale=False), row=i + 1, col=1)

        fig.add_trace(go.Scatter(x=regionLabels, y=structure), row=i + 2, col=1)

        fig.update_layout(
            title_text='FC of simulated signals by stimulation frequency and weight || ' + t_c + ' region: ' + region,
            template="simple_white")
        pio.write_html(fig, file=folder + "/stimSpace-f&a%s_" % t_c + title + ".html", auto_open=auto_open)

    else:

        sp_titles = ["Stimulation Weight = " + str(ws) for ws in set(df.stimAmplitude)]

        fig = make_subplots(rows=len(set(df.stimAmplitude)) + 1, cols=1,
                            subplot_titles=(sp_titles + ["Structural Connectivity"]),
                            shared_yaxes=True, shared_xaxes=True, y_title="Stimulation Frequency")

        for i, ws in enumerate(set(df.stimAmplitude)):
            subset = df[df["stimAmplitude"] == ws]

            if i == 0:
                fig.add_trace(
                    go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                               colorbar=dict(title="PLV")), row=i + 1, col=1)

            else:
                fig.add_trace(
                    go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                               zmin=0, zmax=1, showscale=False), row=i + 1, col=1)

        fig.add_trace(go.Scatter(x=regionLabels, y=structure), row=i + 2, col=1)

        fig.update_layout(
            title_text='FC of simulated signals by stimulation frequency and weight || ' + t_c + ' region: ' + region,
            template="simple_white")
        pio.write_html(fig, file=folder + "/stimSpace-f&a%s_FFTpeaks_" % t_c + title + ".html", auto_open=auto_open)
