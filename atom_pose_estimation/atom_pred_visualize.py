from json.tool import main
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
    dist[dist > 5] = 5 

    diff_x = np.abs(diff[:, 0]); # diff_x[diff_x > 5] = 5
    diff_y = np.abs(diff[:, 1]); # diff_y[diff_y > 5] = 5
    diff_z = np.abs(diff[:, 2]); # diff_z[diff_z > 5] = 5
    _x = np.arange(len(dist))

    fig = plt.figure()
    ax_1 = fig.add_subplot(211)
    ax_21 = fig.add_subplot(234)
    ax_22 = fig.add_subplot(235)
    ax_23 = fig.add_subplot(236)

    if _type == 'line':
        # ax_1.plot(_x, np.abs(diff[:, 0]), marker='+', linestyle=':', markersize=3, label="x-dist",)
        # ax_21.plot(_x, np.abs(diff[:, 1]), marker='x', linestyle=':', markersize=3, label="Axis Y",)
        # ax_22.plot(_x, np.abs(diff[:, 2]), marker='o', linestyle=':', markersize=3, label="Axis Z",)
        # ax_23.plot(_x, np.abs(dist), marker='*', linestyle=':', markersize=3, label="Abs distance",)
        ax_21.plot(_x, np.abs(diff[:, 0]), marker='+', linestyle=':', markersize=3,)
        ax_22.plot(_x, np.abs(diff[:, 1]), marker='x', linestyle=':', markersize=3, )
        ax_23.plot(_x, np.abs(diff[:, 2]), marker='o', linestyle=':', markersize=3, )
        ax_1.plot(_x, np.abs(dist), marker='*', linestyle=':', markersize=3, )
        # ax_1.set_title("Abs distance (A)")
        # ax_21.set_title("x-dist (A)")
        # ax_22.set_title("y-dist (A)")
        # ax_23.set_title("z-dist (A)")
    elif _type == "point":
        ax_21.scatter(_x, diff_x, marker='+', s=5, )
        ax_22.scatter(_x, diff_y, marker='x', s=5, )
        ax_23.scatter(_x, diff_z, marker='o', s=5, )
        ax_1.scatter(_x, np.abs(dist), marker='*', s=5, )
    elif _type == "bin":
        ax_21.hist(diff_x, bins=80)
        ax_22.hist(diff_y, bins=80)
        ax_23.hist(diff_z, bins=80)
        ax_1.hist(np.abs(dist), bins=120)

    ax_1.set_title("Amino-{}: Abs distance (A)".format(target))
    ax_21.set_title("x-dist (A)")
    ax_22.set_title("y-dist (A)")
    ax_23.set_title("z-dist (A)")
    # plt.title(_title)
    # plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "{}_atom_pos_regression_diff.png".format(target)))
    plt.close()


if __name__ == "__main__":
    test_data_1 = np.arange(0, 1000)
    test_data_2 = np.arange(0, 1000, 2)
    test_data_3 = np.arange(0, 1000, 4)
    test_data_4 = np.arange(0, 1000, 5)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.hist(test_data_1, bins=50, )
    ax1.set_title("Ca atom distance", fontsize=16)
    ax_21 = fig.add_subplot(234)
    ax_21.plot(test_data_1, test_data_1, )
    ax_21.set_title("C atom distance", fontsize=12)

    ax_22 = fig.add_subplot(235)
    ax_22.scatter(test_data_1, test_data_1, )
    ax_22.set_title("N atom distance", fontsize=12)

    ax_23 = fig.add_subplot(236)
    ax_23.plot(test_data_1, test_data_1, )
    ax_23.set_title("O atom distance", fontsize=12)

    plt.title("None None")
    plt.tight_layout()

    # a[1][0].hist(test_data_2, bins=50)
    # a[1][1].hist(test_data_3, bins=50)
    # a[1][2].hist(test_data_4, bins=50)
    plt.savefig('./data_temp/temo_ing.jpg')
    plt.close()


