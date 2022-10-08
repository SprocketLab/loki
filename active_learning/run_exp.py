from libs import TreeWithLocus, ActiveSelection, RandomSelection
import copy

def run_active_learning(tree):
    locus_lengths = [len(tree.locus.edges)]
    while len(tree.lambda_) < n_nodes:
        active_selector = ActiveSelection()
        next_node = active_selector.select(tree)
        tree.add_node_to_lambda(next_node)
        tree.update_locus()
        locus_lengths.append(len(tree.locus.edges))
    print('RESULT WITH ACTIVE LEARNING', locus_lengths)
    return locus_lengths

def run_random_selection(tree):
    locus_lengths = [len(tree.locus.edges)]
    while len(tree.lambda_) < n_nodes:
        active_selector = RandomSelection()
        next_node = active_selector.select(tree)
        tree.add_node_to_lambda(next_node)
        tree.update_locus()
        locus_lengths.append(len(tree.locus.edges))
    print('RESULT WITH RANDOM SELECTION', locus_lengths)
    return locus_lengths

if __name__ == '__main__':
    n_nodes = 20
    lambda_init_size = 4
    n_trials = 10
    al_conv_all = []
    rnd_conv_all = []
    for trial in range(n_trials):
        tree = TreeWithLocus(n_nodes, lambda_init_size)
        tree_2 = copy.deepcopy(tree)
        active_learning_convergence = len(run_active_learning(tree))
        random_selection_convergence = len(run_random_selection(tree_2))
        al_conv_all.append(active_learning_convergence)
        rnd_conv_all.append(random_selection_convergence)
    print(al_conv_all)
    print(rnd_conv_all)
    