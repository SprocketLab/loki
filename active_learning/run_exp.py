from libs import TreeWithLocus, ActiveSelection, RandomSelection
import copy

if __name__ == '__main__':
    n_nodes = 7
    lambda_init_size = 3
    tree = TreeWithLocus(n_nodes, lambda_init_size)
    tree_2 = copy.deepcopy(tree)
    locus_lengths = [len(tree.locus.edges)]
    while len(tree.lambda_) < n_nodes:
        active_selector = ActiveSelection()
        next_node = active_selector.select(tree)
        tree.add_node_to_lambda(next_node)
        tree.update_locus()
        locus_lengths.append(len(tree.locus.edges))
    print('RESULT WITH ACTIVE LEARNING', locus_lengths)
    
    
    locus_lengths = [len(tree_2.locus.edges)]
    while len(tree_2.lambda_) < n_nodes:
        active_selector = RandomSelection()
        next_node = active_selector.select(tree_2)
        tree_2.add_node_to_lambda(next_node)
        tree_2.update_locus()
        locus_lengths.append(len(tree_2.locus.edges))
    print('RESULT WITH RANDOM SELECTION', locus_lengths)
