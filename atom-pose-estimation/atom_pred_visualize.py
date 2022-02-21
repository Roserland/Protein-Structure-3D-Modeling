from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def scatter_3D_plot(pred_atoms_pos, gt_atom_pos, target):
    """
    plot 3D scatter plots among space

    """
    pass


def plot_coords_diffs(pred_atoms_pos, gt_atom_pos, target, _type="line",
                     save_dir='./imgs/'):
    """
    plot coords difference among 3 axis of 3D space
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    _title = "Atom {} coordninates differce".format(target)

    diff = np.array(pred_atoms_pos) - np.array(gt_atom_pos)
    # print("{} abs-diff: {}".format(target, np.abs(diff)))
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    _x = np.arange(len(dist))

    if _type == 'line':
        plt.plot(_x, np.abs(diff[:, 0]), marker='+', linestyle=':', markersize=3, label="Axis X",)
        plt.plot(_x, np.abs(diff[:, 1]), marker='x', linestyle=':', markersize=3, label="Axis Y",)
        plt.plot(_x, np.abs(diff[:, 2]), marker='o', linestyle=':', markersize=3, label="Axis Z",)
        plt.plot(_x, np.abs(dist), marker='*', linestyle=':', markersize=3, label="Abs distance",)
    elif _type == "point":
        plt.scatter(_x, np.abs(diff[:, 0]), marker='+', s=5, label="Axis X",)
        plt.scatter(_x, np.abs(diff[:, 1]), marker='x', s=5, label="Axis Y",)
        plt.scatter(_x, np.abs(diff[:, 2]), marker='o', s=5, label="Axis Z",)
        plt.scatter(_x, np.abs(dist), marker='*', s=5, label="Abs distance",)

    plt.title(_title)
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, "{}_atom_pos_regression_diff.png".format(target)))
    plt.close()
    



