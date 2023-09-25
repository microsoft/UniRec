# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from scipy.sparse import csr_matrix
import numba
from unirec.constants.protocols import *
from unirec.utils.file_io import * 
from .ease import EASE

@numba.jit(nopython=True)
def _spase_matrix_mul(a_shape, a_indptr, a_indices, b_data, b_indptr, b_indices, n_users):
    res = np.zeros(a_shape)
    for j in range(n_users):
        a_s, a_e = a_indptr[j], a_indptr[j+1] # All items interacted with user j
        hist = a_indices[a_s: a_e]
        b_s, b_e = b_indptr[j], b_indptr[j+1]
        nz_indices = b_indices[b_s: b_e] #For each user, find users within the batch that are similar to it
        nz_data = b_data[b_s: b_e]
        if len(nz_indices) == 0:
            continue
        for i in hist:
            res[i][nz_indices] += nz_data
    return res.T

class UserCF(EASE):
    def _init_modules(self): 
        self.user_item = None
        self.user_similarity = None

    def solve(self, graph):
        """ Optimize the parameters.
        Args:
            graph(scipy.sparse.spmatrix): the graph of user-item interaction graph, which is a sparse matrix.
        """
        user_degrees = np.squeeze(graph.sum(1).A)
        item_degrees = np.squeeze(graph.sum(0).A)
        edge_weights = np.ones_like(graph.data, dtype=np.float32)
        if self.config['edge_norm'] == EdgeNormType.NONE.value:
            pass
        else:
            i_d = item_degrees[graph.indices]
            edge_weights = edge_weights / i_d
            for i in range(len(graph.indptr)-1):
                s, e = graph.indptr[i], graph.indptr[i+1]
                edge_weights[s:e] = np.sqrt(edge_weights[s:e] / user_degrees[i] + 1e-8)
        A = csr_matrix((edge_weights,
                       graph.indices, 
                       graph.indptr),
                       shape=graph.shape)
        A2 = A.dot(A.T)
        A2.setdiag(0)  ## supress the self to self transition so that there is no information leakage
        self.user_similarity = A2
        self.user_item = graph
        return

    def predict(self, interaction):
        user_id = interaction[ColNames.USERID.value].detach().cpu().numpy()
        item_id = interaction[ColNames.ITEMID.value].detach().cpu().numpy()
        user = self.user_similarity[user_id, :]    # [B, U], sparse
        if len(item_id.shape) == 2: # [B, N], one-vs-k protocol
            scores = [None] * item_id.shape[1]
            for i in range(item_id.shape[1]):
                item = item_id[:, i]
                user2item = self.user_item[:, item].T # [B, U]
                score = (user.multiply(user2item)).sum(-1).A # [B, 1]
                scores[i] = score
            return np.concatenate(scores, axis=1) # [B,N]
        elif len(item_id.shape) == 1: # [B], only one item
            user2item = self.user_item[:, item_id].T # [B, U]
            score = (user.multiply(user2item)).sum(-1).A.squeeze()   # [B]
            return score
        else:
            raise ValueError("`item_id` should be 1/2-dimension.")
    
    def forward_all_item_emb(self, batch_size=None):
        return self.user_item

    def forward_item_emb(self, items, item_features=None):
        return self.user_item.T[items.detach().cpu().numpy(), :]

    def forward_user_emb(self, user_id=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None):
        user = self.user_similarity[:, user_id.detach().cpu().numpy()]    
        return user.T # [B, U], sparse

    def state_dict(self):
        params = {
            "user_similarity": self.user_similarity,
            'user_item': self.user_item
        }
        return params

    def sparse_matrix_mul(self, user_similarity, u_i_graph):
        user_similarity = user_similarity.T
        a_indptr, a_indices = u_i_graph.indptr, u_i_graph.indices
        a_shape = (u_i_graph.shape[1], user_similarity.shape[1]) #(n_items, batch)
        b_data = user_similarity.data
        b_indptr = user_similarity.indptr
        b_indices = user_similarity.indices
        return _spase_matrix_mul(a_shape, a_indptr, a_indices, b_data, b_indptr, b_indices, user_similarity.shape[0])

