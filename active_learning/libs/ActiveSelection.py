import numpy as np
import networkx as nx

class ActiveSelection:
    def __init__(self,):
        pass
    
    def select(self, tree_with_locus):
        '''
        pick lambda_prime that is the furthest from any point in the locus
        (max of min dist from any point in the locus)
        '''
        locus_nodes = tree_with_locus.locus.nodes
        tree_nodes = tree_with_locus.tree.nodes
        non_selected_nodes = np.setxor1d(tree_nodes, locus_nodes)
        shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(tree_with_locus.tree))
        non_selected_nodes_dist = {}
        # print('non selected nodes', non_selected_nodes)
        for node in non_selected_nodes:
            dist_dict = shortest_path_lengths[node]
            min_dist = float("inf")
            min_loc = None
            for loc_node in locus_nodes:
                node_dist_to_loc_node = dist_dict[loc_node]
                if node_dist_to_loc_node < min_dist:
                    min_dist = node_dist_to_loc_node
            non_selected_nodes_dist[node] = min_dist
        next_node = max(non_selected_nodes_dist, key = non_selected_nodes_dist.get) 
        return next_node