
import time
import numpy as np
import scipy.integrate
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline
from plotly.subplots import make_subplots
import plotly.express as px


def timeseries_spectra(signals, simLength, transient, regionLabels, timescale="ms", yaxis="Voltage (mV)",
                       mode="html", param=None, folder="figures", height=400, width=800,
                       freqRange=[2, 40], opacity=1, title="", auto_open=True):

    if "anim" in mode:

        fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], column_widths=[0.7, 0.3], horizontal_spacing=0.15)
        cmap = px.colors.qualitative.Plotly

        timepoints = np.arange(start=transient, stop=simLength, step=len(signals[0][0, :]) / (simLength - transient))

        freqs = np.arange(len(signals[0][0, :]) / 2)
        freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

        # Plot initial traces
        for i, signal in enumerate(signals[0]):
            # Signal
            fig.add_trace(go.Scatter(x=timepoints[:8000], y=signal[:8000], name=regionLabels[i], opacity=opacity,
                                     legendgroup=regionLabels[i], marker_color=cmap[i % len(cmap)]), row=1, col=1)

            # Spectra
            fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
            fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT

            fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies

            fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                     marker_color=cmap[i % len(cmap)], name=regionLabels[i], opacity=opacity,
                                     legendgroup=regionLabels[i], showlegend=False), row=1, col=2)

        # Create frames
        frames, max_power = [], 0
        for i, sim in enumerate(signals):
            t = timescale[i]
            data = []
            for signal in sim:
                # Append signal
                data.append(go.Scatter(y=signal[:8000]))

                # Append spectra
                fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
                fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
                fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies

                data.append(go.Scatter(y=fft))

                max_power = np.max(fft) if np.max(fft) > max_power else max_power

            frames.append(go.Frame(data=data, traces=list(range(len(sim)*2)), name=str(t)))

        fig.update(frames=frames)

        # CONTROLS : Add sliders and buttons
        fig.update_layout(
            xaxis1=dict(title="Time (ms)"), xaxis2=dict(title="Frequency (Hz)"),
            yaxis1=dict(title="Voltage (mV)", range=[np.min(np.asarray(signals)), np.max(np.asarray(signals))]),
            yaxis2=dict(title="Power (dB)", range=[0, max_power]),
            template="plotly_white", title=title, legend=dict(tracegroupgap=2),
            sliders=[dict(
                steps=[dict(method='animate',
                            args=[[str(t)], dict(mode="immediate",
                                                 frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                                 transition=dict(duration=0))],
                            label=str(t)) for i, t in enumerate(timescale)],
                transition=dict(duration=0), x=0.15, xanchor="left", y=-0.15,
                currentvalue=dict(font=dict(size=15), prefix=param + " - ", visible=True, xanchor="right"),
                len=0.8, tickcolor="white")],

            updatemenus=[dict(type="buttons", showactive=False, y=-0.2, x=0, xanchor="left",
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None, dict(frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                                  transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                                  dict(label="Pause", method="animate",
                                       args=[[None],
                                             dict(frame=dict(duration=250, redraw=False, easing="cubic-in-out"),
                                                  transition=dict(duration=0), mode="immediate")])])])

        pio.write_html(fig, file=folder + "/Animated_timeseriesSpectra_" + title + ".html", auto_open=auto_open, auto_play=False)


    else:

        fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], column_widths=[0.7, 0.3], horizontal_spacing=0.15)

        if timescale in ["sec", "seconds", "s", "second"]:
            transient = 1000 * transient
            simLength = 1000 * simLength

        sampling = len(signals[0])/(simLength-transient)  # datapoints/timestep

        timepoints = np.arange(start=transient, stop=simLength, step=1/sampling)

        cmap = px.colors.qualitative.Plotly

        freqs = np.arange(len(signals[0]) / 2)  #
        freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

        for i, signal in enumerate(signals):

            # Timeseries
            if simLength < 8000:
                fig.add_trace(go.Scatter(x=timepoints, y=signal, name=regionLabels[i], opacity=opacity,
                                         legendgroup=regionLabels[i], marker_color=cmap[i%len(cmap)]), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=timepoints[:int(8000*sampling)], y=signal[:int(8000*sampling)], name=regionLabels[i], opacity=opacity,
                                         legendgroup=regionLabels[i], marker_color=cmap[i%len(cmap)]), row=1, col=1)

            # Spectra
            fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
            fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT

            fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies


            fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                     marker_color=cmap[i%len(cmap)], name=regionLabels[i], opacity=opacity,
                                     legendgroup=regionLabels[i], showlegend=False), row=1, col=2)


            fig.update_layout(xaxis=dict(title="Time (ms)"), xaxis2=dict(title="Frequency (Hz)"),
                              yaxis=dict(title=yaxis), yaxis2=dict(title="Power (dB)"),
                              template="plotly_white", title=title, height=height, width=width,
                              legend=dict(orientation="h", y=-0.3, x=0))

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


def timeseries_phaseplane(time, v1, v2, v3=None, mode="html", params=["y0", "y3"], speed=4, folder="figures",
                        opacity=0.7, title="", auto_open=True):

    if len(params) == 3:

        speed = speed if speed >= 1 else 1
        slow = 0 if speed >= 1 else 10 / speed

        fig = make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15],
                            vertical_spacing=0.15,
                            specs=[[{"type":"scene"}], [{}]])

        # Add initial traces: lines
        fig.add_trace(go.Scatter(x=time, y=v1, marker=dict(color="cornflowerblue", opacity=opacity), showlegend=False),
                      row=2, col=1)
        fig.add_trace(
            go.Scatter3d(x=[v2[0]], y=[v1[0]], z=[v3[0]], mode="lines", line=dict(color="cornflowerblue",  width=7),
                         opacity=opacity, showlegend=False), row=1, col=1)

        # Add initial traces: refs
        fig.add_trace(go.Scatter(x=[time[0]], y=[v1[0]], marker=dict(color="red", opacity=opacity), showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter3d(x=[v2[0]], y=[v1[0]], z=[v3[0]], marker=dict(color="red", opacity=opacity, size=5), showlegend=False),
                      row=1, col=1)

        fig.update(frames=[go.Frame(data=[go.Scatter3d(x=v2[:i], y=v1[:i], z=v3[:i]),
                                          go.Scatter(x=[time[i]], y=[v1[i]]),
                                          go.Scatter3d(x=[v2[i]], y=[v1[i]], z=[v3[i]])],
                                    traces=[1, 2, 3], name=str(t)) for i, t in enumerate(time) if
                           (i > 0) & (i % speed == 0)])

        # CONTROLS : Add sliders and buttons
        fig.update_layout(template="plotly_white", height=700, width=700,
                          scene=dict(camera=dict(eye=dict(x=1.25, y=1.25, z=1), center=dict(z=-0.25)),
                              xaxis=dict(title=params[1], range=[min(v2), max(v2)]),
                              yaxis=dict(title=params[0], range=[min(v1), max(v1)]),
                              zaxis=dict(title=params[2], range=[min(v3), max(v3)])),

                          xaxis=dict(title="Time (ms)"), yaxis=dict(title=params[0]),

                          sliders=[dict(
                              steps=[
                                  dict(method='animate',
                                       args=[[str(t)], dict(mode="immediate",
                                                            frame=dict(duration=1 * slow, redraw=True,
                                                                       easing="cubic-in-out"),
                                                            transition=dict(duration=1 * slow))], label=str(t)) for i, t
                                  in
                                  enumerate(time) if (i > 0) & (i % speed == 0)],
                              transition=dict(duration=1 * slow), xanchor="left", x=0.175, y=0.3,
                              currentvalue=dict(font=dict(size=15, color="black"), prefix="Time (ms) - ", visible=True,
                                                xanchor="right"),
                              len=0.7, tickcolor="white", font=dict(color="white"))],

                          updatemenus=[
                              dict(type="buttons", showactive=False, x=0, y=0.25, xanchor="left", direction="left",
                                   buttons=[
                                       dict(label="\u23f5", method="animate",
                                            args=[None,
                                                  dict(frame=dict(duration=1 * slow, redraw=True,
                                                                  easing="cubic-in-out"),
                                                       transition=dict(duration=1 * slow),
                                                       fromcurrent=True, mode='immediate')]),
                                       dict(label="\u23f8", method="animate",
                                            args=[[None],
                                                  dict(frame=dict(duration=1 * slow, redraw=True,
                                                                  easing="cubic-in-out"),
                                                       transition=dict(duration=1 * slow),
                                                       mode="immediate")])])])

        if "html" in mode:
            pio.write_html(fig, file=folder + "/Animated_timeseriesPhasePlane3D_" + title + ".html", auto_open=auto_open,
                           auto_play=False)

        elif "inline" in mode:
            plotly.offline.iplot(fig)

    else:
        speed = speed if speed >= 1 else 1
        slow = 0 if speed >=1 else 10/speed

        fig = make_subplots(rows=2, cols=3, row_heights=[0.8, 0.2],
                            column_widths=[0.2, 0.6, 0.2], vertical_spacing=0.4,
                            specs=[[{}, {}, {}], [{"colspan": 3}, {}, {}]])

        # Add initial traces: lines
        fig.add_trace(go.Scatter(x=time, y=v1, marker=dict(color="cornflowerblue", opacity=opacity), showlegend=False), row=2,
                      col=1)
        fig.add_trace(go.Scatter(x=[v2[0]], y=[v1[0]], marker=dict(color="cornflowerblue", opacity=opacity), showlegend=False),
                      row=1, col=2)

        # Add initial traces: refs
        fig.add_trace(go.Scatter(x=[time[0]], y=[v1[0]], marker=dict(color="red", opacity=opacity), showlegend=False), row=2,
                      col=1)
        fig.add_trace(go.Scatter(x=[v2[0]], y=[v1[0]], marker=dict(color="red", opacity=opacity), showlegend=False), row=1,
                      col=2)

        fig.update(frames=[go.Frame(data=[go.Scatter(x=v2[:i], y=v1[:i]),
                                          go.Scatter(x=[time[i]], y=[v1[i]]),
                                          go.Scatter(x=[v2[i]], y=[v1[i]])],
                                    traces=[1, 2, 3], name=str(t)) for i, t in enumerate(time) if (i > 0) & (i % speed == 0)])

        # CONTROLS : Add sliders and buttons
        fig.update_layout(template="plotly_white", height=600, width=700,
                          xaxis2=dict(title=params[1], range=[min(v2), max(v2)]),
                          yaxis2=dict(title=params[0], range=[min(v1), max(v1)]),
                          xaxis4=dict(title="Time (ms)"), yaxis4=dict(title=params[0]),

                          sliders=[dict(
                              steps=[
                                  dict(method='animate',
                                       args=[[str(t)], dict(mode="immediate",
                                                            frame=dict(duration=1*slow, redraw=False, easing="cubic-in-out"),
                                                            transition=dict(duration=1*slow))], label=str(t)) for i, t in
                                  enumerate(time) if (i > 0) & (i % speed == 0)],
                              transition=dict(duration=1*slow), xanchor="left", x=0.175, y=0.375,
                              currentvalue=dict(font=dict(size=15, color="black"), prefix="Time (ms) - ", visible=True, xanchor="right"),
                              len=0.7, tickcolor="white", font=dict(color="white"))],

                          updatemenus=[dict(type="buttons", showactive=False, x=0, y=0.275, xanchor="left", direction="left",
                                            buttons=[
                                                dict(label="\u23f5", method="animate",
                                                     args=[None,
                                                           dict(frame=dict(duration=1*slow, redraw=False, easing="cubic-in-out"),
                                                                transition=dict(duration=1*slow),
                                                                fromcurrent=True, mode='immediate')]),
                                                dict(label="\u23f8", method="animate",
                                                     args=[[None],
                                                           dict(frame=dict(duration=1*slow, redraw=False, easing="cubic-in-out"),
                                                                transition=dict(duration=1*slow),
                                                                mode="immediate")])])])


        if "html" in mode:
            pio.write_html(fig, file=folder + "/Animated_timeseriesPhasePlane_" + title + ".html", auto_open=auto_open, auto_play=False)

        elif "inline" in mode:
            plotly.offline.iplot(fig)


