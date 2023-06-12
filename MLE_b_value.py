#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:06:55 2023

@author: martin
"""

# Import usual modules
import os
import sys
import datetime
import importlib

import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd

# Import custom style for figures
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(
    "/home/martin/.config/matplotlib/mpl_configdir/stylelib/"
    "BoldAndBeautiful.mplstyle"
)


def MLE_b_value_continuous(
    catalog, threshold_magnitude, delta_magnitude=False
):
    """
    Mle b-value estimation with uncertainty from continuous distribution.

    Estimate b-value from the Magnitudes using a continuous magnitude
    distribution as first described in Utsu (1966) and Aki (1965).

    Sources :
    Marzocchi and Sandri (2003),
    'A review and new insight on the estimation of the b-value and its
    uncertainty.'
    Nava et al (2017),
    'Gutenberg-Richter b-value maximum likelihood estimation and sample size.'
    doi : 10.1007/s10950-016-9589-1
    """
    # If the magnitude precision is not given then use the generic formula
    # given in Marocchi and Sandri 2003 as formula 2.3
    if delta_magnitude is False:
        b_value = (np.log10(np.e)) / (
            (catalog["Magnitude"].mean()) - threshold_magnitude
        )
        b_value_uncertainty = b_value / np.sqrt(len(catalog))

    # If the magnitude is given, use the specific, more precise formula that
    # takes the bias theta2 (from Marzocchi and Sandri, 2003) into account
    # That is, when data is binned, the lowest bin magnitude
    # M_min =M_threshold -delta_magnitude/2
    else:
        b_value = np.log10(np.e) / (
            catalog["Magnitude"].mean()
            - (threshold_magnitude - delta_magnitude / 2)
        )
        b_value_uncertainty = b_value / np.sqrt(len(catalog))

    return b_value, b_value_uncertainty


def MLE_b_value_binned(catalog, completness_magnitude, delta_magnitude=False):
    """
    Mle b-value estimation with uncertainty from binned distribution.

    Estimate b-value from the Magnitudes using a binned magnitude
    distribution as described in Tinti and Mulargia (1987).

    Sources :
    Marzocchi and Sandri (2003),
    'A review and new insight on the estimation of the b-value and its
    uncertainty.'
    """
    p = 1 + (delta_magnitude) / (
        catalog["Magnitude"].mean() - completness_magnitude
    )
    b_value = np.log(p) * np.log10(np.e) / delta_magnitude

    b_value_uncertainty = (1 - p) / (
        np.log(10) * delta_magnitude * np.sqrt(len(catalog) * p)
    )
    return b_value, b_value_uncertainty


# %%
# Synthetic example
from scipy.stats import rv_continuous


def spread_with_sample_size(
    cat, threshold_magnitude, rounding=False, delta_magnitude=False, title=""
):
    plt.figure()
    for idx, sample_size in enumerate([100, 200, 500]):
        b_value_ls = []
        b_value_uncertainty_ls = []
        for i in range(20):
            cat_sample = cat.sample(n=sample_size)
            if rounding:
                cat_sample = rounding * np.round(cat_sample / rounding, 1)
            # Remove magnitudes under the threshold magnitude
            cat_sample_thresh = cat_sample[
                cat_sample["Magnitude"] >= threshold_magnitude
            ]
            b_value, b_value_uncertainty = MLE_b_value_continuous(
                cat_sample_thresh,
                threshold_magnitude=threshold_magnitude,
                delta_magnitude=delta_magnitude,
            )
            b_value_ls.append(b_value)
            b_value_uncertainty_ls.append(b_value_uncertainty)
        plt.axhline(1, c="gray")
        plt.errorbar(
            x=sample_size,
            y=np.mean(b_value_ls),
            yerr=np.std(b_value_ls),
            marker="s",
        )
        plt.xlabel("Sample Size")
        plt.ylabel("b value")
        plt.title(title)
    plt.show()


def mean_magnitude(
    cat,
    threshold_magnitude,
    rounding=False,
    delta_magnitude=False,
):
    plt.figure()
    for idx, sample_size in enumerate([100, 200, 500, 1000]):
        print(sample_size)
        b_value_ls = []
        mean_mag_thresh_ls = []
        mean_mag_ls = []
        b_value_uncertainty_ls = []
        for i in range(20):
            cat_sample = cat.sample(n=sample_size)
            if rounding:
                cat_sample = rounding * np.round(cat_sample / rounding, 1)
            # Remove magnitudes under the threshold magnitude
            cat_sample_thresh = cat_sample[
                cat_sample["Magnitude"] >= threshold_magnitude
            ]
            mean_mag_ls.append(cat_sample["Magnitude"].mean())
            mean_mag_thresh_ls.append(cat_sample_thresh["Magnitude"].mean())

            b_value, b_value_uncertainty = MLE_b_value_continuous(
                cat_sample_thresh,
                threshold_magnitude=threshold_magnitude,
                delta_magnitude=delta_magnitude,
            )
            b_value_ls.append(b_value)
            b_value_uncertainty_ls.append(b_value_uncertainty)

        plt.errorbar(
            x=sample_size,
            y=np.mean(mean_mag_thresh_ls),
            yerr=np.std(mean_mag_thresh_ls),
            marker="o",
            c="C1",
            label=r"$M_{AboveThreshold}$",
        )
        if delta_magnitude:
            plt.errorbar(
                x=sample_size,
                y=np.mean(mean_mag_thresh_ls) - (threshold_magnitude),
                yerr=np.std(
                    [m - threshold_magnitude for m in mean_mag_thresh_ls]
                ),
                marker="+",
                c="C2",
                label=r"$M_{AboveThreshold} - M_{threshold}$",
            )
            plt.errorbar(
                x=sample_size,
                y=np.mean(mean_mag_thresh_ls)
                - (threshold_magnitude - delta_magnitude / 2),
                yerr=np.std(
                    [m - threshold_magnitude for m in mean_mag_thresh_ls]
                ),
                marker="<",
                c="C3",
                label=r"$M_{AboveThreshold} - M_{threshold} + \Delta M/2$",
            )
        plt.xlabel("Sample Size")
        plt.ylabel("Mean magnitude")
        plt.title(
            r"Objective : $M_{mean} - M_{thresh} + \Delta M/2 = M_{mean}$ of continuous data"
        )

        print(cat_sample_thresh)
        if idx == 0:
            plt.legend()
    plt.show()


# %%
class seismicity_catalog(rv_continuous):
    """Power law magnitude distribution."""

    def _pdf(self, M):
        b_truth = 1
        M_min = 0.0
        return b_truth * np.log(10) * 10 ** (-b_truth * (M - M_min))


# a is the lower bound (Mmin) and b the upper bound (at least Mmax)
random_catalog_distribution = seismicity_catalog(a=0.00, b=10, name="catalog")

catalog_size = 10000
random_catalog = pd.DataFrame(
    random_catalog_distribution.rvs(size=catalog_size),
    columns=["Magnitude"],
)
# %%
delta_magnitude = 0.1
rounding = int(delta_magnitude * 10)
threshold_magnitude = 0.1

# %% Mean magnitude

# With the perfect data
mean_magnitude(
    random_catalog,
    threshold_magnitude,
)

# With the rounded data and the original formula
mean_magnitude(
    random_catalog,
    threshold_magnitude,
    rounding=rounding,
)

# With the rounded data and the good formula
mean_magnitude(
    random_catalog,
    threshold_magnitude,
    rounding=rounding,
    delta_magnitude=delta_magnitude,
)

# %% Data spread

# With the perfect data
spread_with_sample_size(
    random_catalog,
    threshold_magnitude,
    title="Continuous magnitudes, continuous formula",
)

# With the rounded data and the original formula
spread_with_sample_size(
    random_catalog,
    threshold_magnitude,
    rounding=rounding,
    title="Rounded magnitudes, continuous formula",
)

# With the rounded data and the good formula
spread_with_sample_size(
    random_catalog,
    threshold_magnitude,
    rounding=rounding,
    delta_magnitude=delta_magnitude,
    title="Rounded magnitudes, binned formula",
)


# %% Figure 5 in Marzocchi et al., 2020


def effect_of_delta_magnitude():
    class seismicity_catalog(rv_continuous):
        """Power law magnitude distribution."""

        def _pdf(self, M):
            b_truth = 1
            M_min = 0.1
            return b_truth * np.log(10) * 10 ** (-b_truth * (M - M_min))

    # a is the lower bound (Mmin) and b the upper bound (at least Mmax)
    random_catalog_distribution = seismicity_catalog(
        a=0.1, b=10, name="catalog"
    )

    catalog_size = 10000
    random_catalog = pd.DataFrame(
        random_catalog_distribution.rvs(size=catalog_size),
        columns=["Magnitude"],
    )
    for rounding in [1, 6]:
        random_catalog_small = random_catalog.sample(n=10000)
        random_catalog_small = rounding * np.round(
            random_catalog_small / rounding, 1
        )
        b = []
        for i in range(1000):
            sample = random_catalog_small.sample(n=100)
            b.append(
                1
                / (
                    (np.log(10))
                    * (
                        np.mean(sample["Magnitude"])
                        - np.min(sample["Magnitude"])
                    )
                )
            )

        plt.hist(b, alpha=0.5)
    plt.show()


# effect_of_delta_magnitude()
