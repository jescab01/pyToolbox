import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots


def fc(df, z=None, title=None, folder="figures", auto_open="True", show_owp=False):

    fig = make_subplots(rows=1, cols=5, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma"),
                        specs=[[{}, {}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                        x_title="Conduction speed (m/s)", y_title="Coupling factor")

    fig.add_trace(go.Heatmap(z=df.Delta, x=df.speed, y=df.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
                             reversescale=True, zmin=-z, zmax=z), row=1, col=1)

    fig.add_trace(go.Heatmap(z=df.Theta, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=df.Alpha, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=3)
    fig.add_trace(go.Heatmap(z=df.Beta, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=4)
    fig.add_trace(go.Heatmap(z=df.Gamma, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=5)

    if show_owp:
        owp_g = df['avg'].idxmax()[0]
        owp_s = df['avg'].idxmax()[1]

        [fig.add_trace(
            go.Scatter(
                x=[owp_s], y=[owp_g], mode='markers',
                marker=dict(color='black', line_width=2, size=[10], opacity=0.5, symbol='circle-open', showscale=False)),
            row=1, col=j) for j in range(1, 6)]

    fig.update_layout(
        title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % title)
    pio.write_html(fig, file=folder + "/paramSpace-g&s_%s.html" % title, auto_open=auto_open)


def significance(df, z=None, title=None, folder="figures", auto_open="True"):
    fig = make_subplots(rows=1, cols=5, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma"),
                            specs=[[{}, {}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                            x_title="Conduction speed (m/s)", y_title="Coupling factor")

    fig.add_trace(
        go.Heatmap(z=df.Delta_sig, x=df.speed, y=df.G, colorscale='RdBu', colorbar=dict(title="Pearson's r p-value"),
                   reversescale=True, zmin=0, zmax=z), row=1, col=1)
    fig.add_trace(
        go.Heatmap(z=df.Theta_sig, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=0, zmax=z,
                   showscale=False), row=1, col=2)
    fig.add_trace(
        go.Heatmap(z=df.Alpha_sig, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=0, zmax=z,
                   showscale=False), row=1, col=3)
    fig.add_trace(
        go.Heatmap(z=df.Beta_sig, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=0, zmax=z,
                   showscale=False), row=1, col=4)
    fig.add_trace(
        go.Heatmap(z=df.Gamma_sig, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=0, zmax=z,
                   showscale=False), row=1, col=5)

    fig.update_layout(
        title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % title)
    pio.write_html(fig, file=folder + "/paramSpace-g&s_%s.html" % title, auto_open=auto_open)