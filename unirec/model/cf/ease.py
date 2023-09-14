import numpy as np
import scipy.sparse as ssp
import numba
from ..base.recommender import BaseRecommender
from unirec.constants.protocols import *


@numba.jit(nopython=True)
def _spase_matrix_mul(a_shape, a_indptr, a_indices, b_data, b_indptr, b_indices):
    res = np.zeros(a_shape)
    for i in range(a_shape[0]):
        s, e = a_indptr[i], a_indptr[i+1]
        hist = a_indices[s:e] # All items interacted with user i
        for j in hist:
            b_s, b_e = b_indptr[j], b_indptr[j+1]
            nz_indices = b_indices[b_s: b_e]    # For each item, find items that are similar to it
            nz_data = b_data[b_s: b_e]
            res[i][nz_indices] += nz_data
    return res

class EASE(BaseRecommender):
    ''' An ItemCF based linear model. 
    EASE is simple linear autoencoder model, whose training objective has a closed-form solution.
    For model details, please refer to https://dl.acm.org/doi/abs/10.1145/3308558.3313710.

    Objective: min_W ||A-AW||^2 + l2_coef * ||W||^2  subject to diag(W)=0
    '''
    # Note: the item_similarity parameters in EASE is a dense matrix of shape (num_item, num_item),
    # which may cause out-of-memory issue in solving. For example, a dataset with 100,000 items will occupy
    # about 40GB * 2 = 80GB memory. Actually, there are some extra data in matrix inversion, so the memory occupied
    # is more than the value.
    # For Taobao dataset with about 80,000 items, 80GB memory is not enough.
    # For dataset with about 10,000 items, 6GB memory is occupied.

    def _init_attributes(self):
        super(EASE, self)._init_attributes()
        # the flag implys that whether the model is optimized by SGD
        self.__optimized_by_SGD__ = False
        self.config_corrector()

    def _init_modules(self):
        self.user_item = None
        self.item_similarity = None

    # def collect_data(self, inter_data, to_device=False, schema={}): 
    #     samples = {
    #         k:inter_data[v].detach().cpu().numpy() for k,v in schema.items()
    #     }
    #     return samples

    def solve(self, graph):
        """ Optimize the parameters.
        Args:
            graph(scipy.sparse.spmatrix): the graph of user-item interaction graph, which is a sparse matrix.
        """
        R = graph
        G = R.T @ R
        diagIndices = np.diag_indices_from(G)
        G[diagIndices] += self.config['l2_coef']
        P = np.linalg.inv(G.todense())
        B = P / (-np.diag(P))
        B[diagIndices] = 0
        self.item_similarity = B
        self.user_item = R
        return
    
    def predict(self, interaction):
        user_id = interaction[ColNames.USERID.value].detach().cpu().numpy()
        item_id = interaction[ColNames.ITEMID.value].detach().cpu().numpy()
        user = self.user_item[user_id, :]    # [B, I], sparse
        if len(item_id.shape) == 2: # [B, N], one-vs-k protocol
            scores = [None] * item_id.shape[1]
            for i in range(item_id.shape[1]):
                item = item_id[:, i]
                item_sim = self.item_similarity[:, item].T # [B, I]
                score = (user.multiply(item_sim)).sum(-1).A # [B, 1]
                scores[i] = score
            return np.concatenate(scores, axis=1) # [B,N]
        elif len(item_id.shape) == 1: # [B], only one item
            item_sim = self.item_similarity[:, item_id].T # [B, I]
            score = (user.multiply(item_sim)).sum(-1).A.squeeze()   # [B]
            return score
        else:
            raise ValueError("`item_id` should be 1/2-dimension.")


    def forward_all_item_emb(self, batch_size=None):
        return self.item_similarity.T

    def forward_item_emb(self, items, item_features=None):
        return self.item_similarity[items.detach().cpu().numpy(), :]

    def forward_user_emb(self, user_id=None, item_seq=None, item_seq_len=None, item_seq_features=None, time_seq=None):
        user = self.user_item[user_id.detach().cpu().numpy(), :]    # [B, I], sparse
        return user 

    def state_dict(self):
        params = {
            "item_similarity": self.item_similarity,
            'user_item': self.user_item
        }
        return params

    def load_state_dict(self, state_dict, strict=False):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def config_corrector(self):
        # Correct the improper configurations. Here `has_user_bias` and `has_item_bias` should be False for all itemcf methods.
        if self.has_user_bias or self.has_item_bias:
            self.logger.info('ItemCF based models do not support user_item_bias, the config is corrected as 0')
            self.has_user_bias = False
            self.has_item_bias = False
            self.config['has_user_bias'] = 0
            self.config['has_item_bias'] = 0
    
    """
    Multiply a user-item graph with item similarity matrix.

    Args:
        u_i_graph: scipy.sparse.csr_matrix. (n_user, n_item). The matrix is 0-1 matrix, which is the key feature contributing to the
                acceleration. The u-i graph is actually the history of users.
        item_similarity: scipy.sparse.csr_matrix. (n_item, n_item). The matrix contains similarity between two items.

    Return:
        np.array: (n_user, n_item). The result of the matrix multiplication.

    """
    def sparse_matrix_mul(self, u_i_graph, item_simarlarity):
        a_indptr, a_indices = u_i_graph.indptr, u_i_graph.indices
        a_shape = u_i_graph.shape 
        b_data = item_simarlarity.data
        b_indptr = item_simarlarity.indptr
        b_indices = item_simarlarity.indices
        return _spase_matrix_mul(a_shape, a_indptr, a_indices, b_data, b_indptr, b_indices)
            