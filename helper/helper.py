import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

# Import ML packs
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC

# General packs
import json

def load_params(file_name: str) -> dict:
    """
    Load parameters from JSON file.

    Params
    ------
    file_name: str
        Path to JSON file

    Return
    ------
    params: dict
        Params encapsulated in a dict.
    """
    
    with open(file_name, "r") as read_json:
        params = json.load(read_json)
    
    return params

def show_scatter(training_points: np.array, title: str, fig_name: str, dot_size: float=60.0, save: bool=False) -> None:
    """
    Simple scatter plot with preset confs.
    
    Params
    ------
    training_points: np.array
        2D array to plot in Euclidean space.
    title: str
        Self explanatory.
    dot_size: float
        The size of the points in the scatter plot.

    Returns
    -------
    None
        Plot the scatter plot.
    """
    
    plt.figure(figsize=(6,6))
    plt.scatter(training_points[:, 0], training_points[:, 1], s=dot_size)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(title)

    if save:
        plt.savefig(r"C:\Users\BV486TN\programs\glasnner_dl\data_figs" + "\\" + fig_name)
    
    plt.show()

def show_clusters(training_points: np.array, fig_name: str, step_mesh_h: float=0.02, sparse_grid: bool=False, dot_size: float=60.0, save: bool=True) -> None:
    """
    Plot clusters with colored background area.
    
    Params
    ------
    training_points: np.array
        2D array to plot in Euclidean space.
    step_mesh_h: float
        The step size to create a mesh from upper and lower bounds
    sparse_grid: bool
        Control if the meshgrid method produces sparse vectors.
    dot_size: float
        The size of the points in the scatter plot.
    
    Returns
    -------
    None
        Plot the cluters with background colors.
    """
    
    # Lower upper bounds in plot axes
    x_min, x_max = training_points[:, 0].min() - 1.0, training_points[:, 0].max() + 1.0
    y_min, y_max = training_points[:, 1].min() - 1.0, training_points[:, 1].max() + 1.0 
    
    # Create a mesh
    mesh_xx, mesh_yy = np.meshgrid(np.arange(x_min, x_max, step_mesh_h), np.arange(y_min, y_max, step_mesh_h))
    
    # Size fig
    plt.figure(figsize=(9, 6))
    
    for cluster_count in range(2, 8):
        # Train predict k-means model
        model_kmeans = KMeans(n_clusters=int(cluster_count))
        model_kmeans.fit(training_points)
        predictions = model_kmeans.predict(training_points)
        
        # Sets fig
        plt.subplot(2, 3, cluster_count - 1)
        plt.xticks([], [])
        plt.yticks([], [])
        
        # Predictions on the right shape for plot
        Z_class = model_kmeans.predict(np.c_[mesh_xx.ravel(), mesh_yy.ravel()])
        Z_class = Z_class.reshape(mesh_xx.shape)
        
        # Plot points and background color
        plt.contourf(mesh_xx, mesh_yy, Z_class, cmap="rainbow", alpha=0.3)
        plt.scatter(training_points[:, 0], training_points[:, 1], s=dot_size, c=predictions, cmap="rainbow")
        plt.title(str(cluster_count) + " Clusters")

    if save:
        plt.savefig(r"C:\Users\BV486TN\programs\glasnner_dl\data_figs" + "\\" + fig_name)
    
    plt.show()

def make_blob_in_box(x_min, x_max, y_min, y_max, num_pts):
    """
    Make data points around a centroid and translate and scale them spatially.
    
    Params
    ------
    x_min: float
        x minimum of the blob
    x_max: float
        x max of the blob
    y_min: float
        y minimum of the blob
    y_max: float
        y max of the blob
    num_pts: int
        
    Return
    ------
    blobxy: list
        list containing the coordinates of the points of the 
    """
    
    blob_xy, bc = make_blobs(n_samples=num_pts, centers=[(0, 0)], n_features=2)
    
    # Extract min and max from blobs
    scale_x_min = np.min(blob_xy[:,0])
    scale_x_max = np.max(blob_xy[:,0])
    scale_y_min = np.min(blob_xy[:,1])
    scale_y_max = np.max(blob_xy[:,1])
    
    # Calculate scale of blobs
    norm_x = [(v[0] - scale_x_min) / (scale_x_max - scale_x_min) for v in blob_xy]
    norm_y = [(v[1] - scale_y_min) / (scale_y_max - scale_y_min) for v in blob_xy]
    
    # Scale blobs
    scale_x = [x_min + ((x_max - x_min) * v) for v in norm_x]
    scale_y = [y_min + ((y_max - y_min) * v) for v in norm_y]
    blobxy = [[scale_x[i], scale_y[i]] for i in range(len(scale_x))]
    
    return blobxy