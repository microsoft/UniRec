# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from scipy.sparse import csr_matrix

from unirec.constants.protocols import *
from unirec.utils.file_io import * 
from .ease import EASE


class SAR(EASE):
    ### For more details about SAR model, please refer to https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/sar_deep_dive.ipynb
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
        A2 = A.T.dot(A)
        A2.setdiag(0)  ## supress the self to self transition so that there is no information leakage
        self.item_similarity = A2
        self.user_item = graph
        return
