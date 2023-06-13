#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:27:38 2023

@author: martin
"""

# Import usual modules
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import rv_continuous

from manage_paths import manage_paths, save_fig_custom
import mle_b_value

# Import custom style for figures
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(
    "/home/martin/.config/matplotlib/mpl_configdir/stylelib/"
    "BoldAndBeautiful.mplstyle"
)


# Get Script Name
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]


# Manage Paths
path_to_figures = manage_paths(SCRIPT_NAME, output=False, figures=True)


def spread_with_sample_size(
    cat,
    threshold_mag,
    rounding=False,
    delta_magnitude=False,
    title="",
    figname=False,
):
    """


    Parameters
    ----------
    cat : TYPE
        DESCRIPTION.
    threshold_mag : TYPE
        DESCRIPTION.
    rounding : TYPE, optional
        DESCRIPTION. The default is False.
    delta_magnitude : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is "".
    figname : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(8, 4))
    for sample_size in [100, 200, 500]:
        b_value_ls = []
        b_value_uncertainty_ls = []
        if rounding:
            cat = rounding * np.round(cat / rounding, 1)
        for _ in range(20):
            # Round the catalog if need
            # Remove magnitudes under the threshold magnitude
            cat_thresh = cat[cat["Magnitude"] >= threshold_mag]
            # Sample the catalog
            cat_sample_thresh = cat_thresh.sample(n=sample_size)
            # Calculate the b-value and uncertainty
            b_value, b_value_uncertainty = mle_b_value.mle_b_value_continuous(
                cat_sample_thresh,
                threshold_magnitude=threshold_mag,
                delta_magnitude=delta_magnitude,
            )
            # Save to lists
            b_value_ls.append(b_value)
            b_value_uncertainty_ls.append(b_value_uncertainty)
        # Plot the true b value
        plt.axhline(1, c="gray")
        # Plot the estimated b value and errors
        plt.errorbar(
            x=sample_size,
            y=np.mean(b_value_ls),
            yerr=np.std(b_value_ls),
            marker="s",
        )
        plt.xlabel("Sample Size")
        plt.ylabel("b value")
        plt.title(title)
    save_fig_custom(path_to_figures, fig, figname)
    plt.show()


def mean_magnitude(
    cat,
    threshold_mag,
    rounding=False,
    delta_magnitude=False,
    figname=False,
):
    """


    Parameters
    ----------
    cat : TYPE
        DESCRIPTION.
    threshold_mag : TYPE
        DESCRIPTION.
    rounding : TYPE, optional
        DESCRIPTION. The default is False.
    delta_magnitude : TYPE, optional
        DESCRIPTION. The default is False.
    figname : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(8, 4))
    for idx, sample_size in enumerate([100, 200, 500, 1000]):
        print(sample_size)
        b_value_ls = []
        mean_mag_thresh_ls = []
        b_value_uncertainty_ls = []
        # Round the catalog samples if asked
        if rounding:
            cat = rounding * np.round(cat / rounding, 1)
        for _ in range(20):
            # Remove magnitudes under the threshold magnitude
            cat_sample_thresh = cat[cat["Magnitude"] >= threshold_mag]
            # Sample
            cat_sample_thresh = cat_sample_thresh.sample(n=sample_size)

            mean_mag_thresh_ls.append(cat_sample_thresh["Magnitude"].mean())

            b_value, b_value_uncertainty = mle_b_value.mle_b_value_continuous(
                cat_sample_thresh,
                threshold_magnitude=threshold_mag,
                delta_magnitude=delta_magnitude,
            )
            b_value_ls.append(b_value)
            b_value_uncertainty_ls.append(b_value_uncertainty)

        plt.errorbar(
            x=sample_size,
            y=np.mean(mean_mag_thresh_ls) - (threshold_mag),
            yerr=np.std([m - threshold_mag for m in mean_mag_thresh_ls]),
            marker="+",
            c="C2",
            label=r"$M_{mean} - M_{threshold}$",
        )
        # If the catalog is
        if delta_magnitude:
            plt.errorbar(
                x=sample_size,
                y=np.mean(mean_mag_thresh_ls)
                - (threshold_mag - delta_magnitude / 2),
                yerr=np.std([m - threshold_mag for m in mean_mag_thresh_ls]),
                marker="<",
                c="C3",
                label=r"$M_{mean} - M_{threshold} + \Delta M/2$",
            )
        plt.xlabel("Sample Size")
        plt.ylabel("Mean magnitude")
        plt.title(
            r"Objective : $M_{mean} - M_{thresh} + \Delta M/2 "
            "= M_{mean} - M_{c}$ of continuous data"
        )
        print(cat_sample_thresh)
        if idx == 0:
            plt.legend(ncol=3)
    save_fig_custom(path_to_figures, fig, figname)
    plt.show()


class SeismicityCatalog(rv_continuous):
    """Power law magnitude distribution."""

    def _pdf(self, x):
        b_truth = 1

        return b_truth * np.log(10) * 10 ** (-b_truth * (x - M_MIN))


def effect_of_delta_magnitude():
    """Figure 5 in Marzocchi et al., 2020."""
    # a is the lower bound (Mmin) and b the upper bound (at least Mmax)
    random_catalog_distribution = SeismicityCatalog(
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
        b_value = []
        for _ in range(1000):
            sample = random_catalog_small.sample(n=100)
            b_value.append(
                1
                / (
                    (np.log(10))
                    * (
                        np.mean(sample["Magnitude"])
                        - np.min(sample["Magnitude"])
                    )
                )
            )

        plt.hist(b_value, alpha=0.5, label=f"Delta_Magnitude_{rounding/10}")
    plt.legend()
    plt.show()


# %%
M_MIN = 0


def main():
    """Run b value evaluation for synthethics."""
    # a is the lower bound (Mmin) and b the upper bound (at least Mmax)
    m_max = 10
    random_catalog_distribution = SeismicityCatalog(
        a=M_MIN, b=m_max, name="catalog"
    )

    catalog_size = 10000
    random_catalog = pd.DataFrame(
        random_catalog_distribution.rvs(size=catalog_size),
        columns=["Magnitude"],
    )
    # %%
    delta_magnitude = 0.1
    rounding = int(delta_magnitude * 10)

    # Threshold magnitude, i.e., magnitude above which the catlaogue is
    # considered complete. When considering truncated magnitudes the
    # completness magnitude may be smaller than the truncation magnitude, e.g.,
    # Mc=0.7, and Mthresh=0.8 with a magnitude delta of 0.1. Indeed, if
    # M_thresh is complete, it is at least complete until the threshold
    # magnitude - deltaM/2. Thus the corrected formula for the binned data.
    completness_magnitude = 0
    # Must be a multiple of delta_magnitude
    threshold_magnitude = 0.5

    # %% Mean magnitude

    # With the perfect data
    mean_magnitude(
        random_catalog,
        completness_magnitude,
        figname="PerfectData",
    )

    # With the perfect data
    spread_with_sample_size(
        random_catalog,
        completness_magnitude,
        title="Continuous magnitudes, continuous formula",
        figname="SampleSize_PerfectData",
    )
    # %%
    # With the rounded data and the original formula
    mean_magnitude(
        random_catalog,
        threshold_magnitude,
        rounding,
        figname="RoundedDataOriginalFormula",
    )
    # With the rounded data and the original formula
    spread_with_sample_size(
        random_catalog,
        threshold_magnitude,
        rounding,
        title="Rounded magnitudes, continuous formula",
        figname="SampleSize_RoundedDataOriginalFormula",
    )

    # %%
    # With the rounded data and the good formula
    mean_magnitude(
        random_catalog,
        threshold_magnitude,
        rounding,
        delta_magnitude=delta_magnitude,
        figname="RoundedDataRoundedFormula",
    )
    # With the rounded data and the good formula
    spread_with_sample_size(
        random_catalog,
        threshold_magnitude,
        rounding,
        delta_magnitude=delta_magnitude,
        title="Rounded magnitudes, binned formula",
        figname="SampleSize_RoundedDataRoundedFormula",
    )
    # %%
    effect_of_delta_magnitude()


if __name__ == "__main__":
    main()
