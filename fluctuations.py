

import os
import time
import glob

import scipy.io
import numpy as np
import pandas as pd

from MFDFA import MFDFA
from mne.time_frequency import tfr_array_morlet

from sklearn.mixture import GaussianMixture
import statsmodels.api as sm

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.signals import epochingTool


"""
Two of functions to perform a fits to data arrays following exponential distributions.
The functions will take the data array, transform, process, fit and will return variables of interest:
R2, AIC, BIC, lambda, etc. 
One function will perform a linear regression over the Log-linear transformation of the data. This fit will be unimodal.
The second function will use a mixture of gaussians to fit in log-log space. This fit will be both unimodal and bimodal.
Optionally, they will show a plot with the fittings.
"""


def LogLikelihoodExp(data, landa, landa2=np.nan, w=1):

    if not np.isnan(landa2):
        pdf_values = w * landa * np.exp(-landa * data) + (1-w) * landa2 * np.exp(-landa2 * data)

    else:
        pdf_values = landa * np.exp(-landa * data)

    log_lkh = np.sum(np.log(pdf_values)) / len(data)

    return log_lkh



def LinearRegression_LogLinear(data, skip_bins=0, plot=False, folder="figures", title="noTitle"):

    # 2. Extract the PDF from histogram
    hist = np.histogram(data, bins=200, density=True)
    # px.scatter(hist[0]).show('browser')

    # 3. Fit Linear Regression on Log-Linear space (remove likelihood==0 bins)
    Y, X = hist[0], (hist[1][:-1] + hist[1][1:]) / 2
    Y, X = Y[Y > 0], X[Y > 0]

    # Define lower likelihood bin - at first iteration will be zero.
    min_lkh = sorted(set(Y))[skip_bins]

    Y_ = np.log(Y[Y >= min_lkh])
    X_ = sm.add_constant(X[Y >= min_lkh])

    model = sm.OLS(Y_, X_)
    model.fit().summary()

    r2adj, pval = model.fit().rsquared_adj, model.fit().pvalues[0]
    constant, landa = model.fit().params[0], -model.fit().params[1]

    # log_lkh, AIC, BIC = model.fit().llf, model.fit().aic, model.fit().bic
    llkh = LogLikelihoodExp(data, landa, landa2=np.nan, w=1)
    AIC = 2 * 1 - 2 * llkh  # AIC=2k−2ln(L)
    BIC = 1 * np.log(len(data)) - 2 * llkh  # BIC=kln(n)−2ln(L)


    # y_pred = model.predict(params=model.fit().params, exog=X_)
    # y_pred = constant - landa * X_[:, 1]

    if plot:
        cmap = px.colors.qualitative.Plotly
        # 4. Plot Log-Linear and Linear-Log spaces
        fig = make_subplots(rows=1, cols=2)

        # 4.1. Log-Linear space + Linear regression
        Y, X = hist[0], (hist[1][:-1] + hist[1][1:]) / 2
        Y, X = np.log(Y[Y > 0]), X[Y > 0]
        y_pred = constant - landa * X
        fig.add_trace(go.Scatter(x=X, y=Y, name="Data", legendgroup="Data", mode="markers", marker=dict(color=cmap[0])), row=1, col=2)
        fig.add_trace(go.Scatter(x=X, y=y_pred, name="Model", legendgroup="Model", line=dict(color=cmap[1])), row=1, col=2)

        # 4.2 Linear-Log space + Transformed distribution
        histlog = np.histogram(np.log(data), bins=200, density=True)
        Y, X = histlog[0], (histlog[1][:-1] + histlog[1][1:]) / 2
        Y, X = Y[Y > 0], X[Y > 0]
        y_pred = landa * np.exp(X - landa * np.exp(X))
        fig.add_trace(go.Scatter(x=X, y=Y, legendgroup="Data", showlegend=False, mode="markers", marker=dict(color=cmap[0])), row=1, col=1)
        fig.add_trace(go.Scatter(x=X, y=y_pred, legendgroup="Model", showlegend=False, line=dict(color=cmap[1])), row=1, col=1)

        fig.update_layout(title="Linear Regression @Log-Linear", template="plotly_white", xaxis2=dict(title="Power"), xaxis=dict(title="log(Power)"),
                          yaxis2=dict(title="log(Likelihood)"), yaxis=dict(title="Likelihood"))

        if plot == "html":
            pio.write_html(fig, file="%s/%s_LR.html" % (folder, title), auto_open=True)

        if plot == "png":
            pio.write_image(fig, file="%s/%s_LR.png" % (folder, title))



    return landa, constant, r2adj, pval, llkh, AIC, BIC



def GaussianMix_LogLog(data, n_components, plot=False, folder="figures", title="noTitle"):

    gmm = GaussianMixture(n_components=n_components, covariance_type='full')

    gmm.fit(data.reshape(-1, 1))

    # log_lkh = gmm.score(data.reshape(-1, 1))


    if n_components == 1:

        landa = 1/gmm.means_.flatten()[0]

        landa2, w = np.nan, np.nan

        if plot:

            hist = np.histogram(np.log(data), bins=200, density=True)
            # px.scatter(hist[0]).show('browser')

            Y, X = hist[0], (hist[1][:-1] + hist[1][1:]) / 2
            Y, X = Y[Y > 0], X[Y > 0]

            cmap = px.colors.qualitative.Plotly
            fig = make_subplots(rows=1, cols=2)
            fig.add_trace(go.Scatter(x=X, y=Y, name="Data", legendgroup="Data", mode="markers", marker=dict(color=cmap[0])), row=1, col=1)

            pdf_unimodal = np.exp(X) * landa * np.exp(-landa * np.exp(X))
            fig.add_trace(go.Scatter(x=X, y=pdf_unimodal, name="Model", legendgroup="Model", line=dict(color=cmap[1])), row=1, col=1)

            hist = np.histogram(data, bins=200, density=True)
            Y, X = hist[0], (hist[1][:-1] + hist[1][1:]) / 2
            Y, X = np.log(Y[Y > 0]), X[Y > 0]
            fig.add_trace(go.Scatter(x=X, y=Y, legendgroup="Data", showlegend=False, mode="markers", marker=dict(color=cmap[0])), row=1, col=2)

            y_pred = np.log(landa) - landa * X  ## constant??
            fig.add_trace(go.Scatter(x=X, y=y_pred, legendgroup="Model", showlegend=False, line=dict(color=cmap[1])), row=1, col=2)

            fig.update_layout(title="GMM unimodal @Linear-Log",template="plotly_white", xaxis=dict(title="log(Power)"), xaxis2=dict(title="Power"),
                              yaxis=dict(title="Likelihood"), yaxis2=dict(title="log(Likelihood)"))

            if plot == "html":
                pio.write_html(fig, file="%s/%s_gmm%i.html" % (folder, title, n_components), auto_open=True)

            if plot == "png":
                pio.write_image(fig, file="%s/%s_gmm%i.png" % (folder, title, n_components))


    elif n_components == 2:

        landa, landa2 = 1 / gmm.means_.flatten()

        w = gmm.weights_[0]

        if plot:

            hist = np.histogram(np.log(data), bins=200, density=True)
            # px.scatter(hist[0]).show('browser')

            Y, X = hist[0], (hist[1][:-1] + hist[1][1:]) / 2
            Y, X = Y[Y > 0], X[Y > 0]

            cmap = px.colors.qualitative.Plotly
            fig = make_subplots(rows=1, cols=2)

            fig.add_trace(go.Scatter(x=X, y=Y, name="Data", legendgroup="Data", mode="markers", marker=dict(color=cmap[0])), row=1, col=1)

            pdf_bimodal = w * np.exp(X) * landa * np.exp(-landa * np.exp(X))
            pdf_bimodal2 = (1 - w) * np.exp(X) * landa2 * np.exp(-landa2 * np.exp(X))
            fig.add_trace(go.Scatter(x=X, y=pdf_bimodal + pdf_bimodal2, name="Model", legendgroup="Model", line=dict(color=cmap[1])), row=1, col=1)
            # fig.add_trace(go.Scatter(x=x, y=pdf_bimodal0, name="Bimodal 0", legendgroup="bimodal0", showlegend=False,
            #                          mode="lines", line=dict(color=cmap[2], width=2)), row=1, col=2)
            # fig.add_trace(go.Scatter(x=x, y=pdf_bimodal1, name="Bimodal 1", legendgroup="bimodal1", showlegend=False,
            #                          mode="lines", line=dict(color=cmap[2], width=2)), row=1, col=2)

            hist = np.histogram(data, bins=200, density=True)
            Y, X = hist[0], (hist[1][:-1] + hist[1][1:]) / 2
            Y, X = np.log(Y[Y > 0]), X[Y > 0]
            fig.add_trace(go.Scatter(x=X, y=Y, legendgroup="Data", showlegend=False, mode="markers", marker=dict(color=cmap[0])), row=1, col=2)

            y_pred = w * landa * np.exp(-landa * X)  # predicting values of the exponential PDFs in linear space; then converting to log.
            y_pred2 = (1-w) * landa2 * np.exp(-landa2 * X)
            fig.add_trace(go.Scatter(x=X, y=np.log(y_pred+y_pred2), legendgroup="Model", showlegend=False, line=dict(color=cmap[1])), row=1, col=2)

            fig.update_layout(title="GMM bimodal @Linear-Log",template="plotly_white", xaxis2=dict(title="Power"), xaxis1=dict(title="log(Power)"),
                              yaxis2=dict(title="log(Likelihood)"), yaxis1=dict(title="Likelihood"))

            if plot == "html":
                pio.write_html(fig, file="%s/%s_gmm%i.html" % (folder, title, n_components), auto_open=True)

            if plot == "png":
                pio.write_image(fig, file="%s/%s_gmm%i.png" % (folder, title, n_components))

    llkh = LogLikelihoodExp(data, landa, landa2, w)

    k = 3 if n_components == 2 else 1
    AIC = 2 * k - 2 * llkh  # AIC = 2 * k − 2 * ln(L)
    BIC = k * np.log(len(data)) - 2 * llkh  # BIC = k * ln(n) − 2 * ln(L)

    return landa, landa2, w,  llkh, AIC, BIC



def ngf(data, samplingFreq, picks=None, lowcut=1, highcut=60, resolution=0.25, plot=False, folder="figures", title="test"):

    if not picks:
        picks = np.arange(0, len(data), 1).tolist()

    data = data[picks, :]

    eSignals = epochingTool(data, 4, samplingFreq, "signals")

    freqs = np.arange(lowcut, highcut, resolution)

    # 2. Compute time-frequency with morlet wavelets. Output (trials, rois, freqs, time)
    tfr = tfr_array_morlet(eSignals, samplingFreq, freqs, n_cycles=7,
                           output="power", zero_mean=True)  ## n_cycles determine the length of the wavelet.

    result = []
    for p, pick in enumerate(picks):

        # Select one source, remove padding from segment and re-concatenate
        tfr_s = np.hstack(tfr[:, p, :, :])  # use only one source per roi

        # CHECK
        # fig = go.Figure(go.Heatmap(z=tfr1[0,:, :40000], x=np.arange(0,40000,1), y=freqs)).show("browser")

        # 3. Get the IAF
        spectrum = np.average(tfr_s, axis=1)
        iaf = freqs[(freqs > 8) & (freqs < 12)][np.argmax(spectrum[(freqs > 8) & (freqs < 12)])]

        # 4. Average the power in alpha band over time
        tfr_iaf = np.average(tfr_s[(iaf - 2 < freqs) & (freqs < iaf + 2), :], axis=0) / np.max(tfr_s)

        # LR output: landa, constant, r2adj, pval, log_lkh, AIC, BIC
        # LR = LinearRegression_LogLinear(tfr_iaf, plot='png', folder=folder, title=title)

        # GMM output: landa, landa2, w, log_lkh, AIC, BIC
        GMM1 = GaussianMix_LogLog(tfr_iaf, 1, plot=plot, folder=folder, title=title)
        GMM2 = GaussianMix_LogLog(tfr_iaf, 2, plot=plot, folder=folder, title=title)

        result.append([pick, GMM1, GMM2])

    return result



def dfa(data, picks=None):

    if not picks:
        picks = np.arange(0, len(data), 1).tolist()


    result = []
    # 1b. Select one channel to process :: channel 64 - MEG1731
    for i, pick in enumerate(picks):

        # Select a band of lags, which usually ranges from
        # very small segments of data, to very long ones, as
        lag = np.unique(2 * np.logspace(0.5, 3, 100).astype(int))
        # Notice these must be ints, since these will segment
        # the data into chucks of lag size


        # Obtain the (MF)DFA
        dfa_lag, dfa_fluct = MFDFA(data[pick, :], lag=lag, q=2, order=1)
        ## q determines what moments of the fluctuation function are being considered.
        # q=2 focuses on the second moment of the fluctuation, and it is the value for standard DFA;
        # q>2 focuses on larger fluctuations (more intense vehaviour) and viceversa for q<2.
        ## Order refers to the polynomial order used to detrending the data. By default, order = 1.


        # 2b. Is a power law relating window size and fluctuations. Therefore,
        # take the logarithm to fit a regression line.
        y, x = np.log(np.squeeze(dfa_fluct)), sm.add_constant(np.log(dfa_lag))
        model = sm.OLS(y, x)
        OLS_result = model.fit()
        # model.fit().summary()

        # pick, OLS.coef, OLS.constant, OLS.r2, OLS.pval
        result.append([pick, OLS_result.params[1], OLS_result.params[0], OLS_result.rsquared_adj, OLS_result.f_pvalue])

    return result