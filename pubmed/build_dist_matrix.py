from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np
import os
from joblib import Parallel, delayed
from multiprocessing import Manager
# import time

class LabelEmbeddingDistanceCalculator:
    def __init__(self):
        pass
    
    def calc_one_row(self, row_i, label_embeddings, dist_obj):
        dists = []
        embedding_i = label_embeddings[row_i, :]
        for row_j in range(label_embeddings.shape[0]):
            if row_j == row_i:
                dists.append(0)
                continue
            embedding_j = label_embeddings[row_j, :]
            cosine_sim_i_j = cosine(embedding_i, embedding_j)
            dists_i_j = np.linalg.norm(embedding_i-embedding_j)
            dists.append(dists_i_j)
        dist_obj.append({
            row_i: dists
        })
    
    def compute(self, label_embeddings):
        manager = Manager()
        dist_obj = manager.list()
        with Parallel(n_jobs=-1) as parallel:
            parallel(delayed(self.calc_one_row)(row_i, label_embeddings, dist_obj) for row_i in tqdm(range(label_embeddings.shape[0])))
        dist_matrix = np.zeros((label_embeddings.shape[0], label_embeddings.shape[0]))
        dist_obj = list(dist_obj)
        for item in dist_obj:
            row_idx = list(item.keys())[0]
            dists = item[row_idx]
            dist_matrix[row_idx, :] = dists
        return dist_matrix

if __name__ == '__main__':
    label_embedding_file = 'features/label.npy'
    store_dir = 'features'
    label_embeddings = np.load(label_embedding_file)
    # start_time = time.time()
    distance_matrix = LabelEmbeddingDistanceCalculator().compute(label_embeddings)
    # print("dim = {} --- %s seconds ---".format(n_dim, time.time() - start_time))
    np.save(os.path.join(store_dir, 'dist_matrix_euc.npy'), distance_matrix)
    np.save(os.path.join(store_dir, 'dist_matrix_euc.npz'), distance_matrix)
    