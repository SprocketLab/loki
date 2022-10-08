import networkx as nx
import numpy as np

class TreeWithLocus:
    def __init__(self, n_nodes, n_init_lambda=3):
        self.tree = nx.random_tree(n=n_nodes)
        print("Generated tree")
        print(nx.forest_str(self.tree, sources=[0]))
        self.set_lambda(sorted(np.random.choice(np.array(self.tree.nodes), n_init_lambda, replace=False)))
        self.update_locus()
        print(f"initial locus: {self.locus.nodes}, length: {len(self.locus.nodes)}")
    
    def get_all_unique_paths_in_lambda(self, lambda_):
        paths = []
        checked = []
        for n1 in lambda_:
            for n2 in lambda_:
                if n1 == n2:
                    continue
                # add another if to see if (n1 n2) has been checked before like (n2 n1)
                if ((n1, n2) in checked) or ((n2, n1) in checked):
                    continue
                checked.append((n1, n2))
                checked.append((n2, n1))
                paths.extend(sorted(nx.all_simple_edge_paths(self.tree, n1, n2)))
        paths.sort(key=len, reverse=True)
        return paths
    
    def create_union_set(self, paths):
        '''
        1. initialize empty graph
        2. sort paths from short to least length
        3. add longest paths first to empty graph
        4. every new addition, check if path exist in graph (can just check from tpl1[0] to tpl-1[1])
        5. add path if not yet in graph
        '''
        G_temp = nx.Graph()
        for i, path in enumerate(paths):
            if G_temp.has_node(path[0][0]) and G_temp.has_node(path[-1][1]):
                if nx.has_path(G_temp, path[0][0], path[-1][1]):
                    continue
            for edge in path:
                G_temp.add_edge(edge[0], edge[1])
        return G_temp
    
    def set_lambda(self, node_list):
        assert isinstance(node_list, list) 
        self.lambda_ = node_list
    
    def add_node_to_lambda(self, node):
        assert node not in self.lambda_
        self.lambda_.append(node)

    def update_locus(self):
        '''
        1. get all unique_paths
        2. create union set
        3. calculate length 
        '''
        unique_paths = self.get_all_unique_paths_in_lambda(self.lambda_)
        self.locus = self.create_union_set(unique_paths)
        self.set_lambda(list(self.locus.nodes))
        return self.locus