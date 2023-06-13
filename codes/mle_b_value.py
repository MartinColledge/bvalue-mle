#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:06:55 2023

@author: martin
"""

# Import usual modules

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import custom style for figures
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(
    "/home/martin/.config/matplotlib/mpl_configdir/stylelib/"
    "BoldAndBeautiful.mplstyle"
)


def mle_b_value_continuous(
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

    Parameters
    ----------
    catalog : pandas DataFrame
        Earthquake catalo.
    threshold_magnitude : float
        Magnitude above which to cut the catalog and under which the catalog
        is considered incomplete.
    delta_magnitude : float, optional
        Precision of magnitudes to which the values are rounded.
        The default is False.

    Returns
    -------
    b_value : float
        GR b value.
    b_value_uncertainty : float
        GR b value uncertainty.

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


def mle_b_value_binned(catalog, completness_magnitude, delta_magnitude=False):
    """
    Mle b-value estimation with uncertainty from binned distribution.

    Estimate b-value from the Magnitudes using a binned magnitude
    distribution as described in Tinti and Mulargia (1987).

    Sources :
    Marzocchi and Sandri (2003),
    'A review and new insight on the estimation of the b-value and its
    uncertainty.'

    Parameters
    ----------
    catalog : pandas DataFrame
        Earthquake catalo.
    threshold_magnitude : float
        Magnitude above which to cut the catalog and under which the catalog
        is considered incomplete.
    delta_magnitude : float, optional
        Precision of magnitudes to which the values are rounded.
        The default is False.

    Returns
    -------
    b_value : float
        GR b value.
    b_value_uncertainty : float
        GR b value uncertainty.

    """
    p_tinti = 1 + (delta_magnitude) / (
        catalog["Magnitude"].mean() - completness_magnitude
    )
    b_value = np.log(p_tinti) * np.log10(np.e) / delta_magnitude

    b_value_uncertainty = (1 - p_tinti) / (
        np.log(10) * delta_magnitude * np.sqrt(len(catalog) * p_tinti)
    )
    return b_value, b_value_uncertainty
