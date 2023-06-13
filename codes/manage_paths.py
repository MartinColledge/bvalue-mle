#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:30:10 2022

@author: martin
"""

import os


def manage_paths(
    script_name,
    output=True,
    figures=True,
    separate_figure_format=["svg", "png", "pdf"],
    videos=False,
):
    """

    Generate and allocate paths to save figures, pipeline outputs or videos

    Parameters
    ----------
    script_name : str
        Name of the script from which this function is called.
    output : bool, optional
        Whether a pipeline output is required. The default is True.
    figures : bool, optional
        Whether a figure folder is required. The default is True.
    separate_figure_format : list, optional
        Which seperate format-specific folders are required for figures.
        The default is ["svg", "png", "pdf"].
    videos : bool, optional
        Whether a video folder is required. The default is False.

    Returns
    -------
    path_to_ouptut : str
        path where reusable output of code is saved to
    path_to_figures : str
        path where output figures of code are saved to

    """

    from pathlib import Path

    # Create list of paths that are created and shall be returned
    returns = []

    if output == True:
        # Get path of data output
        path_to_ouptut = f"../2_pipeline/{script_name}"
        # Create the folder where data is stored if it doesn't exist yet
        Path(path_to_ouptut).mkdir(parents=True, exist_ok=True)
        returns.append(path_to_ouptut)

    if figures == True:
        if type(separate_figure_format) == list:
            paths_to_figures = []
            for formats in separate_figure_format:
                # Get path path of figures ouptut
                path_to_figures = (
                    f"../3_outputs/{script_name}/Figures_{formats}"
                )

                # Create the folder where data is stored if it doesn't exist yet
                Path(path_to_figures).mkdir(parents=True, exist_ok=True)

                # Store this specific format path
                paths_to_figures.append(path_to_figures)

            # Add all the paths to the returns
            returns.append(paths_to_figures)

        else:
            # Get path of figures output
            path_to_figures = f"../3_outputs/{script_name}/Figures"

            # Create the folder where data is stored if it doesn't exist yet
            Path(path_to_figures).mkdir(parents=True, exist_ok=True)
            returns.append([path_to_figures])

    if videos == True:
        # Get path path of video ouptut
        path_to_videos = f"../3_outputs/{script_name}/videos"
        # Create the folder where data is stored if it doesn't exist yet
        Path(path_to_videos).mkdir(parents=True, exist_ok=True)
        returns.append(path_to_videos)

    if len(returns) == 1:
        return returns[0]
    else:
        return returns


def get_file_names(
    paths_to_data,
    file_extensions,
    path_number=None,
    start_with="",
    end_with="",
):
    # If there is only one path for the data, it won't be specified
    if path_number == None:
        # If you it is given as a list, get the first and only element
        if len(paths_to_data) == 1:
            paths_to_data = paths_to_data[0]
        # If you it is given as a list, get the first and only element
        if len(file_extensions) == 1:
            file_extensions = file_extensions[0]
        # Get the files in the path that fit the conditions
        files = [
            f
            for f in os.listdir(paths_to_data)
            if f.endswith(file_extensions)
            and not f.startswith(".")
            and f.startswith(start_with)
            and f.endswith(end_with)
        ]
        # If there is only one file return it
        if len(files) == 1:
            files = files[0]
        return paths_to_data, file_extensions, files
    # If there are multiple paths go to the one that corresponds to the number
    else:
        files = [
            f
            for f in os.listdir(paths_to_data[path_number])
            if f.endswith(file_extensions[path_number])
            and not f.startswith(".")
            and f.startswith(start_with)
            and f.endswith(end_with)
        ]
        # Return the relevant path, extension and files
        return paths_to_data[path_number], file_extensions[path_number], files


def save_df(df, path_to_output, short_name):
    """

    Save a DataFrame to the parquet format

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to save.
    path_to_output : string
        Path where the file should be saved.
    short_name : string
        Name of the file that should be saved.

    Returns
    -------
    None.

    """
    df.to_parquet(f"{path_to_output}/{short_name}.parquet")


def save_fig_custom(path_to_figures, fig, figname, formats=["svg", "png"]):
    for figure_extension in formats:
        # Get the path that contains the format
        path = [p for p in path_to_figures if figure_extension in p]
        fig.savefig(f"{path[0]}/{figname}.{figure_extension}")
