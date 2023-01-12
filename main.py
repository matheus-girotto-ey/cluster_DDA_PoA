# Import scientific packs
import numpy as np
import matplotlib
from helper.helper import *


if __name__ == "__main__":

    # Make plots look nice 
    matplotlib.rcParams["axes.titlesize"]  = 14
    matplotlib.rcParams["axes.labelsize"]  = 14
    matplotlib.rcParams["axes.labelpad"]   = 8
    matplotlib.rcParams["xtick.labelsize"] = 14
    matplotlib.rcParams["ytick.labelsize"] = 14

    # Load parameters from json
    params = load_params("params.json")

    np.random.seed(params["random_seed"])
    dot_size = params["dot_size"]
    num_pts = params["num_pts"]

    # Make synthetic data
    bxy1 = make_blob_in_box(-4, -1, -4, 0, num_pts)
    bxy2 = make_blob_in_box(-4,  0,  2,  4, num_pts)
    bxy3 = make_blob_in_box(0, 1.5, -1.5,  0, num_pts)
    bxy4 = make_blob_in_box(2.5, 4, 0, 3, num_pts)
    bxy5 = make_blob_in_box(2.5, 4, -4, -3, num_pts)
    training_points = np.vstack( [ bxy1, bxy2, bxy3, bxy4, bxy5 ] )

    # Print plots to screen
    show_scatter(training_points, title='Unclassified Cluster', save=True, fig_name="bare_points.jpg")
    show_clusters(training_points, save=True, fig_name="colored_cluster.jpg")