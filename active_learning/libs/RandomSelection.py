import numpy as np

class RandomSelection:
    def __init__(self,):
        pass
    
    def select(self, tree_with_locus):
        locus_nodes = tree_with_locus.locus.nodes
        tree_nodes = tree_with_locus.tree.nodes
        non_selected_nodes = np.setxor1d(tree_nodes, locus_nodes)
        next_node = np.random.choice(non_selected_nodes)
        return next_node