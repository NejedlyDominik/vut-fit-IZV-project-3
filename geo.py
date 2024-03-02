#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np
import errno
import os
# muzeze pridat vlastni knihovny


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Convert given dataframe to geopandas.GeoDataFrame with the correct encoding (EPSG:5514 - Krovak's projection).

    Arguments:
        df (pd.DataFrame)    - dataframe of accidents data

    Return value:
        geopandas.GeoDataFrame    - accidents data with 'geometry' column specifying locations of accidents
    """
    df = df[df['d'].notna() & df['e'].notna()]    # remove all records with unknown location

    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['d'], df['e']), crs='EPSG:5514')


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """
    Plot a graph with two subgraphs according to the location type of the accident in the selected region.

    Arguments:
        gdf (geopandas.GeoDataFrame)    - accidents data with correctly specified locations of accidents
        fig_location (str, optional)    - address, where the figure is stored (default None - figure is not saved)
        show_figure (bool, optional)    - if is set, the figure is shown in the window (default False)
    """
    if(fig_location and show_figure == False):
        # there is no need to do anything, if both of these arguments are not set
        return

    # get records of selected region
    region = 'MSK'
    gdf = gdf[gdf['region'] == region].copy()

    # set encoding suitable for visualisation
    gdf = gdf.to_crs(epsg=3857)

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, constrained_layout=True,  figsize=(13.4, 6))
    fig.suptitle(f'Accidents in the {region} region', fontsize=16, fontweight='bold')

    # plot locations of accidents to the subgraphs accoding to the location type
    gdf[gdf['p5a'] == 1].plot(ax=axes[0], markersize=1, color='tab:red')
    gdf[gdf['p5a'] == 2].plot(ax=axes[1], markersize=1, color='tab:orange')

    # customize individual subplots
    for ax, where in zip(axes, ['In', 'Outside']):
        ax.set_title(f'{where} the village', fontsize=14)

        __customize_subplot(ax, gdf.crs.to_string())

    if fig_location:
        __save_image(fig, fig_location)

    if(show_figure):
        plt.show()


def __customize_subplot(ax, used_crs):
    """
    Set the specific appearance to subplot.

    Arguments:
        ax         - subplot to customisation
        used_crs   - encoding of GeodataFrame
    """
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    # set limits to eliminate incorrect data
    ax.set_xlim([1.9e6, 2.11e6])
    ax.set_ylim([6.33e6, 6.51e6])

    # add base map to subplot
    ctx.add_basemap(ax, crs=used_crs, source=ctx.providers.Stamen.TonerLite)

    ax.spines['top'].set_color('#DADADA')
    ax.spines['bottom'].set_color('#DADADA')
    ax.spines['left'].set_color('#DADADA')
    ax.spines['right'].set_color('#DADADA')


def __save_image(fig, fig_location):
    """
    Save figure to the specified location.

    Arguments:
        fig             - image to be saved
        fig_location    - address, where the figure is stored
    """
    fig_location = os.path.normpath(fig_location)
    dir_name, fig_file = os.path.split(fig_location)

    # create directories
    if dir_name != '':
            try:
                os.makedirs(dir_name)
            except OSError as e:
                if(e.errno != errno.EEXIST):
                    raise

    fig.savefig(fig_location)


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Plot a graph with locations of all accidents in the selected region clustered into clusters.

    Args:
        gdf (geopandas.GeoDataFrame)    - accidents data with correctly specified locations of accidents
        fig_location (str, optional)    - address, where the figure is stored (default None - figure is not saved)
        show_figure (bool, optional)    - if is set, the figure is shown in the window (default False)
    """
    if(fig_location and show_figure == False):
        # there is no need to do anything, if both of these arguments are not set
        return

    # get records of selected region
    region = 'MSK'
    gdf = gdf[gdf['region'] == region].copy()

    # set encoding suitable for visualisation
    gdf = gdf.to_crs(epsg=3857)

    # categorize accidents into clusters
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=13, max_iter=500, n_init=20).fit(np.column_stack([gdf.geometry.x, gdf.geometry.y]))

    # add column with clusters
    gdf['cluster'] = kmeans.labels_

    # get numbers of accidents in individual clusters
    gdf = gdf.dissolve(by="cluster", aggfunc={'region': 'count'}).rename(columns={'region': "cnt"})

    # add geometry of clusters to the dataframe and set geometry to accidents locations
    gdf = gdf.merge(geopandas.GeoDataFrame(geometry=geopandas.points_from_xy(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])), left_on="cluster", right_index=True).set_geometry('geometry_x')

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True,  figsize=(8, 6))
    ax.set_title(f'Accidents in the {region} region', fontsize=14)

    # plot locations of accidents to the subplot
    gdf.plot(ax=ax, markersize=1, color='tab:grey')

    # set geometry to clusters
    gdf = gdf.set_geometry('geometry_y')

    # plot clusters to the subplot and add color bar, which is set to opaque
    gdf.plot(ax=ax, markersize=gdf['cnt'] / 4, column='cnt', alpha=0.5, vmin=0, legend=True)

    __customize_subplot(ax, gdf.crs.to_string())

    if fig_location:
        __save_image(fig, fig_location)

    if(show_figure):
        plt.show()


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
