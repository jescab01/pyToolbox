
import numpy as np
from itertools import combinations, chain

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

from tvb.simulator.lab import connectivity
import nibabel as nib
import shutil
from templateflow import api as tflow
import json
from skimage import measure, segmentation

# data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
#
# subj = "HC-fam"
# conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
# conn.weights = conn.scaled_weights(mode="tract")
#
# networks = [
#     ("Perception Network",
#      ['Left-Amygdala', 'Right-Amygdala', 'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal',
#       'ctx-lh-fusiform', 'ctx-rh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
#       'ctx-lh-lateraloccipital', 'ctx-rh-lateraloccipital',
#       'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-precentral', 'ctx-rh-precentral',
#       'ctx-lh-superiorparietal', 'ctx-rh-superiorparietal', 'ctx-lh-insula', 'ctx-rh-insula']),
#
#     ("Mirror Network",
#      ['ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
#       'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal',
#       'ctx-lh-lateraloccipital', 'ctx-rh-lateraloccipital',
#       'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-precentral', 'ctx-rh-precentral',
#       'ctx-lh-superiorparietal', 'ctx-rh-superiorparietal', 'ctx-lh-insula', 'ctx-rh-insula']),
#
#     ("Emotional Network",
#      ['Left-Thalamus', 'Right-Thalamus', 'ctx-lh-entorhinal', 'ctx-rh-entorhinal',
#       'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate', 'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate',
#
#       'Left-Caudate', 'Right-Caudate', 'Left-Putamen', 'Right-Putamen', 'Left-Pallidum', 'Right-Pallidum',
#       'Left-Amygdala', 'Right-Amygdala', 'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',
#       'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal',
#       'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate', 'ctx-lh-insula', 'ctx-rh-insula']),
#
#     ("Mentalizing Network",
#      ['ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',
#       'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal',
#       'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-rh-parsorbitalis',
#       'ctx-lh-parstriangularis', 'ctx-rh-parstriangularis',
#       'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate', 'ctx-lh-precuneus', 'ctx-rh-precuneus',
#       'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
#       'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'ctx-lh-supramarginal', 'ctx-rh-supramarginal',
#       'ctx-lh-temporalpole', 'ctx-rh-temporalpole'])]
#
# cortical_rois = ['ctx-lh-bankssts', 'ctx-rh-bankssts',
#                  'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',
#                  'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-rh-cuneus',
#                  'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
#                  'ctx-lh-fusiform', 'ctx-rh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
#                  'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal', 'ctx-lh-insula', 'ctx-rh-insula',
#                  'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate', 'ctx-lh-lateraloccipital',
#                  'ctx-rh-lateraloccipital',
#                  'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-rh-lingual',
#                  'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-middletemporal',
#                  'ctx-rh-middletemporal',
#                  'ctx-lh-paracentral', 'ctx-rh-paracentral', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
#                  'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-rh-parsorbitalis',
#                  'ctx-lh-parstriangularis', 'ctx-rh-parstriangularis', 'ctx-lh-pericalcarine',
#                  'ctx-rh-pericalcarine',
#                  'ctx-lh-postcentral', 'ctx-rh-postcentral', 'ctx-lh-posteriorcingulate',
#                  'ctx-rh-posteriorcingulate',
#                  'ctx-lh-precentral', 'ctx-rh-precentral', 'ctx-lh-precuneus', 'ctx-rh-precuneus',
#                  'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
#                  'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
#                  'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal', 'ctx-lh-superiorparietal',
#                  'ctx-rh-superiorparietal',
#                  'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'ctx-lh-supramarginal',
#                  'ctx-rh-supramarginal',
#                  'ctx-lh-temporalpole', 'ctx-rh-temporalpole', 'ctx-lh-transversetemporal',
#                  'ctx-rh-transversetemporal']
#
# ## Better if you arrange your conn data intercalating left and right rois
# l = [roi for roi in conn.region_labels if ("Left" in roi) or ("ctx-lh" in roi)]
# r = [roi for roi in conn.region_labels if ("Right" in roi) or ("ctx-rh" in roi)]
#
# all_rois = list(chain(*zip(l, r)))
#
# # Sort SC labels.
# SClabs = list(conn.region_labels)
# SC_idx = [SClabs.index(roi) for roi in all_rois]
#
# conn.region_labels = conn.region_labels[SC_idx]
# conn.weights = conn.weights[:, SC_idx][SC_idx]
# conn.centres = conn.centres[SC_idx, :]
#
# threshold = 0.05
#

################# Template transparent brain
def addpial(fig, mode, row=None, col=None, opacity=0.2, showlegend=True, corr_x=0, corr_y=2, corr_z=-5):
    # Load template data
    shutil.copy(
        "E:\LCCN_Local\PycharmProjects\\toolbox\\littlebrains_template-tpl-MNI152NLin2009cAsym_res-01_desc-brain_probseg.nii.gz",
        "C:/Users/jescab01/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_probseg.nii.gz")

    template = tflow.get(template='MNI152NLin2009cAsym', desc="brain", resolution=1, suffix="probseg")
    img = nib.load(template)
    data = img.get_fdata()

    affine = img.affine[:-1, -1]
    affine[0] += corr_x
    affine[1] += corr_y
    affine[2] += corr_z

    #########  GLASS APPROACH
    if "glass" in mode:
        # Load template settings
        netplotbrainpath = "C:\\Users\jescab01\AppData\Local\Programs\Python\Python39\Lib\site-packages\\netplotbrain"
        with open(netplotbrainpath + '/templatesettings/templatesettings_glass.json', 'r') as f:
            glass_kwargs_all = json.load(f)

        if template in glass_kwargs_all:
            glass_kwargs = glass_kwargs_all[template]
        else:
            glass_kwargs = glass_kwargs_all['default']

        # perform segmentation.
        segments = segmentation.slic(data, glass_kwargs['template_glass_nsegments'],
                                     compactness=glass_kwargs['template_glass_compactness'],
                                     enforce_connectivity=False,
                                     start_label=1,
                                     channel_axis=None,
                                     min_size_factor=glass_kwargs['template_glass_minsizefactor'],
                                     max_size_factor=glass_kwargs['template_glass_maxsizefactor'])

        borders = segmentation.find_boundaries(segments, mode='thick')
        # Scale the alpha of the border values based on template intensity
        data[~borders] = 0
        points = np.where(data != 0)

        points = points + affine

        skip = 5
        if row is None:
            fig.add_trace(go.Scatter3d(x=points[0][::skip], y=points[1][::skip], z=points[2][::skip], hoverinfo="skip",
                                       opacity=glass_kwargs['template_glass_maxalpha'], showlegend=showlegend, name="Pial", legendgroup="Pial",
                                       marker=dict(opacity=glass_kwargs['template_glass_maxalpha'],
                                                   size=glass_kwargs['template_glass_pointsize'], color="gray")))
        else:
            fig.add_trace(go.Scatter3d(x=points[0][::skip], y=points[1][::skip], z=points[2][::skip], hoverinfo="skip",
                                       opacity=glass_kwargs['template_glass_maxalpha'], showlegend=showlegend, name="Pial", legendgroup="Pial",
                                       marker=dict(opacity=glass_kwargs['template_glass_maxalpha'],
                                                   size=glass_kwargs['template_glass_pointsize'], color="gray")),
                          row=row, col=col)

    ##### SURFACE APPROACH
    else:
        # detect surface using skimage's marching cubes
        verts, faces, normals, _ = measure.marching_cubes(
            data, level=None, step_size=2)

        verts = verts + affine

        if row is None:
            fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                    opacity=opacity, color="lightgrey", hoverinfo="skip", showlegend=showlegend, name="Pial", legendgroup="Pial",
                                    lighting=dict(diffuse=0.8, fresnel=0.2, roughness=0.2, specular=0.25)))

        else:
            fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                    opacity=opacity, color="lightgrey", hoverinfo="skip", showlegend=showlegend, name="Pial", legendgroup="Pial",
                                    lighting=dict(diffuse=0.8, fresnel=0.2, roughness=0.2, specular=0.25)), row=row,
                          col=col)

    return fig


def plot_littlebrains(conn, threshold=0.1, networks=None,
                      mode="single", pialmode="surface", title="", folder="figures"):
    """
    For a better plotting experience, make sure the nodes are sorted intercalated:
    i.e. roi1_lh, roi1_rh, roi2_lr, ...

    :param conn: tvb type connectivity object
    :param threshold: SC will be normalized and under-threshold connections will be discarded.
    :param networks:
    :param mode: if networks, ["single" (one brain with inserted networks) xor "multiple"
    (adds to "simple, rows with 3 views of netwoks networks)].
    :param pialmode: ["surface" (2D covering and continuous sheet) xor "glass" (smoked transparent points)].
    :param title:
    :param folder: Where to save output figures (.html and .svg)
    :return:
    """

    conn.weights = conn.scaled_weights("tract")

    #########################
    ## First approach, plot simply the brain with nodes in color (one hemisphere darker than the other),
    # and degree in size; edges in black.
    if networks is None:

        fig = go.Figure()

        # Edges trace
        ## Filter edges to show: remove low connected nodes via thresholding
        edges_ids = list(combinations([i for i, roi in enumerate(conn.region_labels)], r=2))
        edges_ids = [(i, j) for i, j in edges_ids if conn.weights[i, j] > threshold]

        ## Define [start, end, None] per coordinate and connection
        edges_x = [elem for sublist in [[conn.centres[i, 0]] + [conn.centres[j, 0]] + [None] for i, j in edges_ids] for
                   elem in sublist]
        edges_y = [elem for sublist in [[conn.centres[i, 1]] + [conn.centres[j, 1]] + [None] for i, j in edges_ids] for
                   elem in sublist]
        edges_z = [elem for sublist in [[conn.centres[i, 2]] + [conn.centres[j, 2]] + [None] for i, j in edges_ids] for
                   elem in sublist]

        ## Define color per connection based on FC changes
        fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip",
                                   line=dict(color="gray", width=4), opacity=0.5, name="Edges"))

        # Nodes trace
        ## Define size per degree
        degree, size_range = np.sum(conn.weights, axis=1), [8, 30]
        size = ((degree - np.min(degree)) * (size_range[1] - size_range[0]) / (np.max(degree) - np.min(degree))) + \
               size_range[0]

        ## Define color per node: Distrubute categorical colours
        nodes_color = [px.colors.sample_colorscale("Jet", i / (len(conn.region_labels) + 1))[0] for i, roi in
                       enumerate(conn.region_labels)]
        #
        # px.colors.sample_colorscale("Viridis", 0.9)
        # px.colors.qualitative.Alphabet

        # Create text labels per ROI
        hovertext3d = ["<b>" + roi + "</b><br>Degree " + str(round(degree[i], 5))
                        for i, roi in enumerate(conn.region_labels)]

        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d, mode="markers", name="Nodes",
                                   marker=dict(size=size, opacity=1, color=nodes_color,
                                               line=dict(color="gray", width=2))))

        # Add brain surface
        fig = addpial(fig, pialmode)

        fig.update_layout(
            template="plotly_white", legend=dict(x=0.8, y=0.5),
            scene=dict(xaxis=dict(title="Sagital axis<br>(R-L)"),
                       yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                       zaxis=dict(title="Horizontal axis<br>(sup-inf)")))

    #######################################
    ## Second approach, plot several networks in single brain.
    elif networks and "single" in mode:

        fig = go.Figure()

        # Edges trace
        ## Filter edges to show: remove low connected nodes via thresholding
        edges_ids = list(combinations([i for i, roi in enumerate(conn.region_labels)], r=2))
        edges_ids = [(i, j) for i, j in edges_ids if conn.weights[i, j] > threshold[0]]

        ## Define [start, end, None] per coordinate and connection
        edges_x = [elem for sublist in [[conn.centres[i, 0]] + [conn.centres[j, 0]] + [None] for i, j in edges_ids] for
                   elem in sublist]
        edges_y = [elem for sublist in [[conn.centres[i, 1]] + [conn.centres[j, 1]] + [None] for i, j in edges_ids] for
                   elem in sublist]
        edges_z = [elem for sublist in [[conn.centres[i, 2]] + [conn.centres[j, 2]] + [None] for i, j in edges_ids] for
                   elem in sublist]

        ## Baseline trace
        fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip",
                                   line=dict(color="lightgray", width=4), opacity=0.2, name="Edges",
                                   legendgroup="Edges"))

        ## Network traces
        ### Define color per connection based on Networks
        cmap = px.colors.qualitative.Pastel2
        for c, network in enumerate(networks):
            name, rois = network

            SClabs = list(conn.region_labels)
            SC_idx = [SClabs.index(roi) for roi in rois]

            edges_ids = list(combinations(SC_idx, r=2))
            edges_ids = [(i, j) for i, j in edges_ids if conn.weights[i, j] > threshold[c + 1]]

            ## Define [start, end, None] per coordinate and connection
            edges_x = [elem for sublist in [[conn.centres[i, 0]] + [conn.centres[j, 0]] + [None] for i, j in edges_ids]
                       for elem in sublist]
            edges_y = [elem for sublist in [[conn.centres[i, 1]] + [conn.centres[j, 1]] + [None] for i, j in edges_ids]
                       for elem in sublist]
            edges_z = [elem for sublist in [[conn.centres[i, 2]] + [conn.centres[j, 2]] + [None] for i, j in edges_ids]
                       for elem in sublist]

            ## Baseline trace
            fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip",
                                       line=dict(color=cmap[c], width=4), opacity=0.8, name="Edges: " + name,
                                       legendgroup="Edges"))

        # Nodes trace
        ## Define size per degree
        degree, size_range = np.sum(conn.weights, axis=1), [8, 30]
        size = ((degree - np.min(degree)) * (size_range[1] - size_range[0]) / (np.max(degree) - np.min(degree))) + \
               size_range[0]

        ## Define color per node: Distrubute categorical colours
        nodes_color = [px.colors.sample_colorscale("Jet", i / (len(conn.region_labels) + 1))[0] for i, roi in
                       enumerate(conn.region_labels)]

        # Create text labels per ROI
        hovertext3d = ["<b>" + roi + "</b><br>Degree " + str(round(degree[i], 5))
                        for i, roi in enumerate(conn.region_labels)]

        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d, mode="markers", name="Nodes",
                                   marker=dict(size=size, opacity=1, color=nodes_color,
                                               line=dict(color="gray", width=2))))

        # Add brain surface
        fig = addpial(fig, pialmode)

        fig.update_layout(
            template="plotly_white", legend=dict(x=0.8, y=0.5, groupclick="toggleitem"),
            scene=dict(xaxis=dict(title="Sagital axis<br>(R-L)"),
                       yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                       zaxis=dict(title="Horizontal axis<br>(sup-inf)")))

    #######################################
    ## Third approach, plot several networks in multiple subplots.
    elif networks and "multiple" in mode:

        specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene", "rowspan":4}]] + [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {}] for row in range(len(networks)-1)]
        fig = make_subplots(rows=len(networks), cols=4, specs=specs, column_widths=[0.15, 0.15, 0.15, 0.55],
                            horizontal_spacing=0, vertical_spacing=0.01)

        ### GENERAL NETWORK
        # Edges trace
        ## Filter edges to show: remove low connected nodes via thresholding
        edges_ids = list(combinations([i for i, roi in enumerate(conn.region_labels)], r=2))
        edges_ids = [(i, j) for i, j in edges_ids if conn.weights[i, j] > threshold[0]]

        ## Define [start, end, None] per coordinate and connection
        edges_x = [elem for sublist in [[conn.centres[i, 0]] + [conn.centres[j, 0]] + [None] for i, j in edges_ids] for
                   elem in sublist]
        edges_y = [elem for sublist in [[conn.centres[i, 1]] + [conn.centres[j, 1]] + [None] for i, j in edges_ids] for
                   elem in sublist]
        edges_z = [elem for sublist in [[conn.centres[i, 2]] + [conn.centres[j, 2]] + [None] for i, j in edges_ids] for
                   elem in sublist]

        ## Baseline trace
        fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip",
                                   line=dict(color="lightgray", width=4), opacity=0.2, legendgroup="Edges", name="Edges"), row=1, col=4)

        ## Network traces
        ### Define color per connection based on Networks
        cmap = px.colors.qualitative.Pastel2
        for c, network in enumerate(networks):
            name, rois = network

            SClabs = list(conn.region_labels)
            SC_idx = [SClabs.index(roi) for roi in rois]

            edges_ids = list(combinations(SC_idx, r=2))
            edges_ids = [(i, j) for i, j in edges_ids if conn.weights[i, j] > threshold[c + 1]]

            ## Define [start, end, None] per coordinate and connection
            edges_x = [elem for sublist in [[conn.centres[i, 0]] + [conn.centres[j, 0]] + [None] for i, j in edges_ids]
                       for elem in sublist]
            edges_y = [elem for sublist in [[conn.centres[i, 1]] + [conn.centres[j, 1]] + [None] for i, j in edges_ids]
                       for elem in sublist]
            edges_z = [elem for sublist in [[conn.centres[i, 2]] + [conn.centres[j, 2]] + [None] for i, j in edges_ids]
                       for elem in sublist]

            ## Baseline trace
            fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip",
                                       line=dict(color=cmap[c], width=4), opacity=0.8, name=name, legendgroup=name), row=1, col=4)

        # Nodes trace
        ## Define size per degree
        degree, size_range = np.sum(conn.weights, axis=1), [8, 30]
        size = ((degree - np.min(degree)) * (size_range[1] - size_range[0]) / (np.max(degree) - np.min(degree))) + \
               size_range[0]

        ## Define color per node: Distrubute categorical colours
        nodes_color = [px.colors.sample_colorscale("Jet", i / (len(conn.region_labels) + 1))[0] for i, roi in
                       enumerate(conn.region_labels)]

        # Create text labels per ROI
        hovertext3d = ["<b>" + roi + "</b><br>Degree " + str(round(degree[i], 5))
                        for i, roi in enumerate(conn.region_labels)]

        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d, mode="markers", name="Nodes", legendgroup="Nodes",
                                   marker=dict(size=size, opacity=1, color=nodes_color,
                                               line=dict(color="gray", width=2))), row=1, col=4)

        # Add brain surface
        fig = addpial(fig, pialmode, row=1, col=4)


        ### LITTLE NETWORKS
        scaling = 0.5
        cam_dist = 1.5
        camera_lateral = dict(eye=dict(x=cam_dist, y=0, z=0))
        camera_frontal = dict(eye=dict(x=0, y=cam_dist, z=0))
        camera_superior = dict(eye=dict(x=0, y=0, z=cam_dist), up=dict(x=0, y=-1, z=0))

        for c, network in enumerate(networks):
            name, rois = network

            SClabs = list(conn.region_labels)
            SC_idx = [SClabs.index(roi) for roi in rois]

            edges_ids = list(combinations(SC_idx, r=2))
            edges_ids = [(i, j) for i, j in edges_ids if conn.weights[i, j] > threshold[c + 1]]

            ## Define [start, end, None] per coordinate and connection
            edges_x = [elem for sublist in [[conn.centres[i, 0]] + [conn.centres[j, 0]] + [None] for i, j in edges_ids]
                       for elem in sublist]
            edges_y = [elem for sublist in [[conn.centres[i, 1]] + [conn.centres[j, 1]] + [None] for i, j in edges_ids]
                       for elem in sublist]
            edges_z = [elem for sublist in [[conn.centres[i, 2]] + [conn.centres[j, 2]] + [None] for i, j in edges_ids]
                       for elem in sublist]

            ## Baseline trace
            fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip", showlegend=False,
                                       line=dict(color=cmap[c], width=4), opacity=0.8, legendgroup=name), row=1 + c, col=1)

            fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip", showlegend=False,
                                       line=dict(color=cmap[c], width=4), opacity=0.8, legendgroup=name), row=1 + c, col=2)

            fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip", showlegend=False,
                                       line=dict(color=cmap[c], width=4), opacity=0.8, legendgroup=name), row=1 + c, col=3)

            # Nodes trace
            fig.add_trace(go.Scatter3d(x=conn.centres[SC_idx, 0], y=conn.centres[SC_idx, 1], z=conn.centres[SC_idx, 2], hoverinfo="text",
                                       hovertext=np.array(hovertext3d)[SC_idx], mode="markers", legendgroup="Nodes", showlegend=False,
                                       marker=dict(size=size*scaling, opacity=1, color=np.array(nodes_color)[SC_idx],
                                                   line=dict(color="gray", width=2))), row=1 + c, col=1)

            fig.add_trace(go.Scatter3d(x=conn.centres[SC_idx, 0], y=conn.centres[SC_idx, 1], z=conn.centres[SC_idx, 2], hoverinfo="text",
                                       hovertext=np.array(hovertext3d)[SC_idx], mode="markers", legendgroup="Nodes", showlegend=False,
                                       marker=dict(size=size*scaling, opacity=1, color=np.array(nodes_color)[SC_idx],
                                                   line=dict(color="gray", width=2))), row=1 + c, col=2)

            fig.add_trace(go.Scatter3d(x=conn.centres[SC_idx, 0], y=conn.centres[SC_idx, 1], z=conn.centres[SC_idx, 2], hoverinfo="text",
                                       hovertext=np.array(hovertext3d)[SC_idx], mode="markers", legendgroup="Nodes", showlegend=False,
                                       marker=dict(size=size*scaling, opacity=1, color=np.array(nodes_color)[SC_idx],
                                                   line=dict(color="gray", width=2))), row=1 + c, col=3)

            # Add brain surface
            fig = addpial(fig, pialmode, row=1 + c, col=1, opacity=0.1, showlegend=False)
            fig = addpial(fig, pialmode, row=1 + c, col=2, opacity=0.15, showlegend=False)
            fig = addpial(fig, pialmode, row=1 + c, col=3, opacity=0.1, showlegend=False)

        fig.update_layout(
            template="plotly_white", legend=dict(x=1, y=0.5, groupclick="toggleitem"),
            scene1=dict(camera=camera_lateral, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene2=dict(camera=camera_frontal, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene3=dict(camera=camera_superior, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene4=dict(camera=dict(eye=dict(x=1.1, y=1.75, z=0.5)),
                        xaxis=dict(title="Sagital axis<br>(R-L)"),
                        yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                        zaxis=dict(title="Horizontal axis<br>(sup-inf)")),
            scene5=dict(camera=camera_lateral, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene6=dict(camera=camera_frontal, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene7=dict(camera=camera_superior, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene8=dict(camera=camera_lateral, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene9=dict(camera=camera_frontal, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene10=dict(camera=camera_superior, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene11=dict(camera=camera_lateral, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene12=dict(camera=camera_frontal, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            scene13=dict(camera=camera_superior, xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),)

    pio.write_html(fig, file=folder + "/3DLittleBrains_" + mode + title + ".html", auto_open=True)
    pio.write_image(fig, file=folder + "/3DLittleBrains_" + mode + title + ".svg", height=700, width=1300)


# plot_littlebrains(conn, [0.1, 0.01, 0.01, 0.01, 0.01], networks, mode="multiple", pialmode="surface", title="", folder="figures")
