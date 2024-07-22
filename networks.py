
import time
import numpy as np
import networkx as nx
from itertools import chain

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# TOPOLOGICAL FILTERING
def omst_filter(matrix, mode="optimize", maxiter=15, verbose=True):
    """
    This algorithm is a method proposed by [Dimitriadis et al. (2017)](https://doi.org/10.3389/fninf.2017.00028)
    to prune functional connectivity brain networks and leave the most basic structural connections.
    Two types of rules are usually used to do this task: thresholding and topological.
    Thresholding rules are those based on the absolute weight, on the proportion, on the mean degree of the network.
    Instead, topological algorithms try to respect the underlying architecture of the network.
    [Jiang et al. 2023](https://doi.org/10.1162/netn_a_00315) found this method to be the most reliable one
    among others used to date.

    :param matrix:
    :param mode: "debug" if you want to iterate a specific number of times
    :param maxiter: Number of iterations in the debug mode
    :return: omst filtered matrix (outputG) + information about algorithm's behaviour ()
    """

    # Creates the input graph
    inputG = nx.from_numpy_array(matrix)
    input_degree = sum(dict(inputG.degree(weight="weight")).values())

    # Preallocate output_graph
    outputG = nx.from_numpy_array(np.zeros(shape=matrix.shape))
    output_degree = sum(dict(outputG.degree(weight="weight")).values())

    ## OMST ALGORITHM
    quality = nx.global_efficiency(outputG) - output_degree / input_degree
    new_quality, iter, tic0 = 0, 0, time.time()

    qualities, eff, cost, msts, out_t, in_t = [], [], [], [], [], []

    iter, maxiter = (0, 0) if mode != "debug" else (0, maxiter)
    while (new_quality >= quality and len(nx.minimum_spanning_tree(inputG).edges) > 0) or (iter < maxiter):

        tic = time.time()

        in_t.append(inputG.copy())
        out_t.append(outputG.copy())

        mstG = nx.minimum_spanning_tree(inputG)
        msts.append(mstG.copy())

        temp_edges = []
        # while new_quality >= quality and len(mstG.edges) > 0:
        while len(mstG.edges) > 0:
            quality = new_quality
            qualities.append(new_quality)  # for debugging purposes

            # Sort MST edges by betweeness
            # lambda use _Defines the sorting value with repect to x (i.e., each iteration value from .items() - key[0], value[1])
            edge_betweeness_mst = sorted(nx.edge_betweenness_centrality(mstG).items(), key=lambda x: x[1], reverse=True)

            (temp_u, temp_v), weight = edge_betweeness_mst[0]

            # Add best edge to OMST
            outputG.add_edge(temp_u, temp_v, weight=inputG[temp_u][temp_v]["weight"])
            # Remove edge from MST
            mstG.remove_edge(temp_u, temp_v)

            temp_edges.append(edge_betweeness_mst[0][0])

            # Calculate new quality
            output_degree = sum(dict(outputG.degree(weight="weight")).values())
            new_quality = nx.global_efficiency(outputG) - output_degree / input_degree

            eff.append(nx.global_efficiency(outputG))
            cost.append(output_degree / input_degree)

            if verbose:
                print(
                    "Iteration %i.  omstG edges %i/%i  Quality = %0.5f  |  \u0394(quality) = %0.10f  (%0.2f/%0.2f s)  - Go!"
                    % (iter, len(outputG.edges), len(inputG.edges), quality, new_quality - quality, time.time() - tic,
                       time.time() - tic0), end="\r")

        # Remove edges added to omstG from inputG
        for temp_u, temp_v in temp_edges:
            inputG.remove_edge(temp_u, temp_v)

        iter += 1

    print("Iteration %i.  omstG edges %i/%i  Quality = %0.5f  |  \u0394(quality) = %0.10f  (%0.2f/%0.2f s)  - Done!"
          % (iter, len(outputG.edges), len(inputG.edges), quality, new_quality - quality, time.time() - tic,
             time.time() - tic0))

    return outputG, [qualities, eff, cost, in_t, msts, out_t]


# interactive VIZUALIZATION of NETWORKS
def network_viz(graphs, node_attrs=None, types=None, layout="spring", dim=2, animated=False, title="test", folder="figures"):
    """

    :param graphs: list of graphs separating graph types with n_graphs inside. n_types of graphs will define the number
    of columns in the figure. For static graphs (i.e., animated=False) only one graph per type is needed
    (e.g., [[graph1]]), in case of more graphs passed only the first one will be plotted. More than one graph per
    type could be used for building animations (e.g., [[graph1, ..., graph_n], [graph1, ..., graph_n], ...]).

    :param node_attrs: Nodes' id, names, symbol (hemisphere), group(lobe), colors.
    :param types:
    :param layout: Algorithm applied for the spatial arrangement of nodes: kamada_kaway xor spring (i.e., Fruchterman-Reingold).
    :param dim: Dimensions for the vizualization ( 2D / 3D )
    :param animated:
    :param title:
    :param folder:
    :return:
    """


    cols, pos_nodes = len(graphs), []
    specs = [[{} for i in range(cols)]] if dim == 2 else [[{"type": "surface"} for i in range(cols)]]

    fig = make_subplots(rows=1, cols=cols, specs=specs, column_titles=types)

    # The static graph is the first frame of the animation.
    # It must be built anyway.
    for i in range(cols):

        # Extract the network: as it is static, I just need the first xor the only passed graph
        graph_temp = graphs[i][0]

        # Determine the position of nodes
        pos_nodes.append([nx.kamada_kawai_layout(graph_temp, weight="weight", dim=dim) if "kamada" in layout\
            else nx.spring_layout(graph_temp, weight="weight", dim=dim)])

        # degree_nodes = np.array([degree + 0.5 for node, degree in graph_temp.degree])

        sl = True if i == 0 else False

        # Build the graph, different dimensions different building
        if dim == 2:

            pos_nodes_array = np.array([[key, value[0], value[1]] for key, value in pos_nodes[i][-1].items()])

            # Edges array structure: From (node, x, y); To   (node, x, y);  (nan, nan, nan);  ...
            pos_edges_array = [[[frm] + list(pos_nodes[i][-1][frm]), [to] + list(pos_nodes[i][-1][to]), [None, None, None]] for
                               frm, to in graph_temp.edges() if frm != to]
            pos_edges_array = np.array(list(chain.from_iterable(pos_edges_array)))

            # width_edges = [[[frm, to, weight], [None, None, None]] for
            #                    frm, to, weight in graph_temp.edges.data("weight") if frm != to]
            # width_edges = np.array(list(chain.from_iterable(width_edges)))

            # Add first edge traces to maintain them in the background
            if pos_edges_array.any():
                fig.add_trace(go.Scatter(x=pos_edges_array[:, 1], y=pos_edges_array[:, 2], showlegend=sl, name="edges", legendgroup="edges",
                                         line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'), row=1, col=1+i)
            else:
                fig.add_trace(go.Scatter(x=None, y=None, showlegend=sl, name="edges",
                                         legendgroup="edges", line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'), row=1,
                              col=1 + i)
            ## Add nodes
            for symbol in set(node_attrs["symbol"]):
                for group in set(node_attrs["group"]):
                    sub_ids = node_attrs["id"].loc[(node_attrs["symbol"] == symbol) & (node_attrs["group"] == group)].values
                    fig.add_trace(go.Scatter(
                        x=pos_nodes_array[sub_ids, 1], y=pos_nodes_array[sub_ids, 2],
                        mode='markers', hoverinfo="text", hovertext=node_attrs["name"].values[sub_ids],
                        showlegend=sl, name=group, legendgroup=group,
                        marker=dict(colorscale='Turbo', reversescale=True, color=node_attrs["color"].values[sub_ids],
                                    cmin=0, cmax=4, size=10, line_width=2, symbol=symbol)),
                        row=1, col=1+i)

        elif dim == 3:

            pos_nodes_array = np.array([[key, value[0], value[1], value[2]] for key, value in pos_nodes[i][-1].items()])

            # Edges array structure: From (node, x, y, z); To   (node, x, y, z);  (nan, nan, nan, nan);  ...
            pos_edges_array = [[[frm] + list(pos_nodes[i][-1][frm]), [to] + list(pos_nodes[i][-1][to]), [None, None, None, None]] for
                               frm, to in graph_temp.edges() if frm != to]
            pos_edges_array = np.array(list(chain.from_iterable(pos_edges_array)))

            # Add first edge traces to maintain them in the background
            if pos_edges_array.any():
                fig.add_trace(go.Scatter3d(x=pos_edges_array[:, 1], y=pos_edges_array[:, 2], z=pos_edges_array[:, 3],
                                           showlegend=sl, name="edges", legendgroup="edges",
                                           line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'), row=1, col=1+i)
            else:
                fig.add_trace(go.Scatter3d(x=None, y=None, z=None,
                                           showlegend=sl, name="edges", legendgroup="edges",
                                           line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'), row=1, col=1+i)
            ## Add nodes
            for symbol in set(node_attrs["symbol"]):
                for group in set(node_attrs["group"]):
                    sub_ids = node_attrs["id"].loc[
                        (node_attrs["symbol"] == symbol) & (node_attrs["group"] == group)].values
                    fig.add_trace(go.Scatter3d(
                        x=pos_nodes_array[sub_ids, 1], y=pos_nodes_array[sub_ids, 2], z=pos_nodes_array[sub_ids, 3],
                        mode='markers', hoverinfo="text", hovertext=node_attrs["name"].values[sub_ids],
                        showlegend=sl, name=group, legendgroup=group,
                        marker=dict(colorscale='Turbo', reversescale=True, color=node_attrs["color"].values[sub_ids],
                                    cmin=0, cmax=4, size=10, line_width=2, symbol=symbol)),
                        row=1, col=1+i)

    fig.update_layout(template="plotly_white")

    if animated:

        if dim == 2:

            # You have to mount a set of frames defining what traces to change and how.
            nframes = len(graphs[0])

            ## Now add frames with data
            frames = []
            for frame in range(nframes):

                frame_traces = []
                for g_type in range(len(graphs)):
                    graph_temp = graphs[g_type][frame]

                    # Calculate positions based on previous ones for smooth transitions;
                    pos_nodes[g_type].append(nx.kamada_kawai_layout(graph_temp, weight="weight", dim=dim,
                                                      pos=pos_nodes[g_type][-1]) if "kamada" in layout \
                        else nx.spring_layout(graph_temp, weight="weight", dim=dim, pos=pos_nodes[g_type][-1]))

                    pos_nodes_array = np.array([[key, value[0], value[1]] for key, value in pos_nodes[g_type][-1].items()])

                    # Edges array structure: From (node, x, y); To   (node, x, y);  (nan, nan, nan);  ...
                    pos_edges_array = [[[frm] + list(pos_nodes[g_type][-1][frm]), [to] + list(pos_nodes[g_type][-1][to]), [None, None, None]]
                                       for frm, to in graph_temp.edges() if frm != to]
                    pos_edges_array = np.array(list(chain.from_iterable(pos_edges_array)))


                    # update EDGES position and presence
                    if pos_edges_array.any():
                        frame_traces.append(go.Scatter(x=pos_edges_array[:, 1], y=pos_edges_array[:, 2]))
                    else:
                        frame_traces.append(go.Scatter(x=None, y=None))

                    # update NODES position
                    for symbol in set(node_attrs["symbol"]):
                        for group in set(node_attrs["group"]):
                            sub_ids = node_attrs["id"].loc[
                                (node_attrs["symbol"] == symbol) & (node_attrs["group"] == group)].values

                            frame_traces.append(go.Scatter(x=pos_nodes_array[sub_ids, 1], y=pos_nodes_array[sub_ids, 2]))

                frames.append(
                    go.Frame(data=frame_traces,
                             traces=list(range(cols + cols * len(set(node_attrs["symbol"])) * len(set(node_attrs["group"])))),
                             name=str(frame)))

            fig.update(frames=frames)


            # CONTROLS : Add sliders and buttons
            fig.update_layout(
                sliders=[dict(
                    steps=[
                        dict(method='animate', args=[[str(frame)],
                                                     dict(mode="immediate", frame=dict(duration=4000),
                                                          transition=dict(duration=3000, redraw=False, easing="cubic-in-out"))], label=str(frame))
                        for frame in range(nframes)],
                    transition=dict(duration=0), x=0.1, xanchor="left", y=-0.1,
                    currentvalue=dict(font=dict(size=15), prefix="Iteration - ", visible=True, xanchor="right"),
                    len=0.52, tickcolor="white")],

                updatemenus=[dict(type="buttons", showactive=False, y=-0.15, x=0, xanchor="left",
                                  buttons=[
                                      dict(label="Play", method="animate",
                                           args=[None,
                                                 dict(frame=dict(duration=4000),
                                                      transition=dict(duration=3000, redraw=False, easing="cubic-in-out"),
                                                      fromcurrent=True, mode='immediate')]),
                                      dict(label="Pause", method="animate",
                                           args=[[None],
                                                 dict(frame=dict(duration=4000),
                                                      transition=dict(duration=3000, redraw=False, easing="cubic-in-out"),
                                                      mode="immediate")])])])


        # Usa las posiciones previas para ir transicionando el layout.
        if dim == 3:

            # You have to mount a set of frames defining what traces to change and how.
            nframes = len(graphs[0])

            frames = []
            for frame in range(nframes):

                frame_traces = []
                for g_type in range(len(graphs)):
                    graph_temp = graphs[g_type][frame]

                    # Calculate positions based on previous ones for smooth transitions;
                    pos_nodes[g_type].append(nx.kamada_kawai_layout(graph_temp, weight="weight", dim=dim,
                                                                    pos=pos_nodes[g_type][-1]) if "kamada" in layout \
                                                 else nx.spring_layout(graph_temp, weight="weight", dim=dim,
                                                                       pos=pos_nodes[g_type][-1]))

                    pos_nodes_array = np.array([[key, value[0], value[1], value[2]] for key, value in pos_nodes[g_type][-1].items()])

                    # Edges array structure: From (node, x, y, z); To   (node, x, y, z);  (nan, nan, nan, nan);  ...
                    pos_edges_array = [[[frm] + list(pos_nodes[g_type][-1][frm]), [to] + list(pos_nodes[g_type][-1][to]), [None, None, None, None]]
                                       for frm, to in graph_temp.edges() if frm != to]
                    pos_edges_array = np.array(list(chain.from_iterable(pos_edges_array)))


                    # update EDGES position and presence
                    if pos_edges_array.any():
                        frame_traces.append(go.Scatter3d(x=pos_edges_array[:, 1], y=pos_edges_array[:, 2], z=pos_edges_array[:, 3]))
                    else:
                        frame_traces.append(go.Scatter3d(x=None, y=None, z=None))

                    # update NODES position
                    for symbol in set(node_attrs["symbol"]):
                        for group in set(node_attrs["group"]):
                            sub_ids = node_attrs["id"].loc[
                                (node_attrs["symbol"] == symbol) & (node_attrs["group"] == group)].values

                            frame_traces.append(go.Scatter3d(x=pos_nodes_array[sub_ids, 1], y=pos_nodes_array[sub_ids, 2], z=pos_nodes_array[sub_ids, 3]))

                frames.append(
                    go.Frame(data=frame_traces,
                             traces=list(range(cols + cols * len(set(node_attrs["symbol"])) * len(set(node_attrs["group"])))),
                             name=str(frame)))

            fig.update(frames=frames)


            # CONTROLS : Add sliders and buttons
            fig.update_layout(
                sliders=[dict(
                    steps=[
                        dict(method='animate', args=[[str(frame)],
                                                     dict(mode="immediate", frame=dict(duration=2000),
                                                          transition=dict(duration=2000, redraw=False, easing="cubic-in-out"))], label=str(frame))
                        for frame in range(nframes)],
                    transition=dict(duration=0), x=0.1, xanchor="left", y=-0.1,
                    currentvalue=dict(font=dict(size=15), prefix="Iteration - ", visible=True, xanchor="right"),
                    len=0.52, tickcolor="white")],

                updatemenus=[dict(type="buttons", showactive=False, y=-0.15, x=0, xanchor="left",
                                  buttons=[
                                      dict(label="Play", method="animate",
                                           args=[None,
                                                 dict(frame=dict(duration=2000),
                                                      transition=dict(duration=2000, redraw=False, easing="cubic-in-out"),
                                                      fromcurrent=True, mode='immediate')]),
                                      dict(label="Pause", method="animate",
                                           args=[[None],
                                                 dict(frame=dict(duration=2000),
                                                      transition=dict(duration=2000, redraw=False, easing="cubic-in-out"),
                                                      mode="immediate")])])])

    pio.write_html(fig, file=folder + "/network" + str(dim) + "D_" + title + ".html", auto_open=True, auto_play=False)

    return

