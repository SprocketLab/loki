from libs import TreeWithLocus, ActiveSelection, RandomSelection
import copy
from tqdm import tqdm
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_active_learning(tree):
    locus_lengths = [len(tree.locus.nodes)]
    while len(tree.lambda_) < n_nodes:
        active_selector = ActiveSelection()
        next_node = active_selector.select(tree)
        tree.add_node_to_lambda(next_node)
        tree.update_locus()
        locus_lengths.append(len(tree.locus.nodes))
    while len(locus_lengths) <= n_nodes - lambda_init_size:
        locus_lengths.append(locus_lengths[-1])
    return locus_lengths


def run_random_selection(tree):
    locus_lengths = [len(tree.locus.nodes)]
    while len(tree.lambda_) < n_nodes:
        active_selector = RandomSelection()
        next_node = active_selector.select(tree)
        tree.add_node_to_lambda(next_node)
        tree.update_locus()
        locus_lengths.append(len(tree.locus.nodes))
    while len(locus_lengths) <= n_nodes - lambda_init_size:
        locus_lengths.append(locus_lengths[-1])
    return locus_lengths


def plot_results(active_all, rnd_all):
    active_all = np.vstack(active_all)
    rnd_all = np.vstack(rnd_all)

    active_all_mean = np.mean(active_all, axis=0)
    rnd_all_mean = np.mean(rnd_all, axis=0)

    active_all_uq = np.quantile(active_all, 0.75, axis=0)
    active_all_lq = np.quantile(active_all, 0.25, axis=0)
    rnd_all_uq = np.quantile(rnd_all, 0.75, axis=0)
    rnd_all_lq = np.quantile(rnd_all, 0.25, axis=0)

    ticksize = 12
    plt.rcParams["figure.figsize"] = (5, 3.5)
    # plt.rcParams["xtick.labelsize"] = ticksize
    # plt.rcParams["ytick.labelsize"] = ticksize
    linewidth = 4
    plt.fill_between(
        np.arange(len(active_all_mean)),
        active_all_lq,
        active_all_uq,
        linewidth=1,
        alpha=0.2,
    )
    plt.plot(active_all_mean, label="Active learning", linewidth=linewidth, alpha=0.75)
    plt.fill_between(
        np.arange(len(rnd_all_mean)),
        rnd_all_lq,
        rnd_all_uq,
        linewidth=1,
        alpha=0.2,
    )
    plt.plot(rnd_all_mean, label="Random", linewidth=linewidth, alpha=0.75)
    plt.ylabel(r"Locus size")
    plt.xlabel("# of selections")
    xticks = [i - lambda_init_size for i in range(lambda_init_size, n_nodes + 1)]
    #plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    #plt.xticks(range(0, len(xticks)), xticks, rotation=45)
    plt.tight_layout()
    plt.legend(frameon=False)
    # plt.grid()
    plt.title("Active next class selection")
    plt.tight_layout()
    #plt.savefig(f"results/n_{n_nodes}lambda_{lambda_init_size}.pdf")
    plt.savefig(f"results/active.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n_nodes", type=int, help="number of tree nodes", required=True
    )
    parser.add_argument(
        "-l", "--lambda_size", type=int, help="initial lambda size", required=True
    )
    parser.add_argument(
        "-n_rep", "--n_repeat", type=int, help="number of repetitions", required=True
    )
    args = parser.parse_args()

    n_nodes = args.n_nodes
    lambda_init_size = args.lambda_size
    n_trials = args.n_repeat
    active_conv_all = []
    rnd_conv_all = []
    for trial in tqdm(range(n_trials)):
        tree = TreeWithLocus(n_nodes, lambda_init_size)
        tree_2 = copy.deepcopy(tree)
        active_learning_convergence = run_active_learning(tree)
        random_selection_convergence = run_random_selection(tree_2)
        active_conv_all.append(active_learning_convergence)
        rnd_conv_all.append(random_selection_convergence)
    plot_results(active_conv_all, rnd_conv_all)
