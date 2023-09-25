# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
import scipy.sparse as ssp
from tqdm import tqdm

from .ease import EASE

'''
Reference: 
    SLIM: Sparse Linear Methods for Top-N Recommender Systems (https://ieeexplore.ieee.org/document/6137254)
    Objective: 
        min_W  1/2 ||A-AW||^2 + l1_coef * ||W||_1 + l1_coef * ||W||^2
        subject to W >= 0 and diag(W)=0, where A is the adjacency matrix of user-item interactions
'''


class SLIM(EASE):
    def solve(self, graph, verbose=2):
        """ Optimize the parameters.
        Args:
            graph(scipy.sparse.spmatrix): the graph of user-item interaction graph, which is a sparse matrix.
            verbose (int): Control wether to show the progress of training epoch and evaluate epoch. 
                            0: show nothing;  1: show basic progress message, but no tqdm progress bar; 2: show everything
        """
        # Note: the iteration procedure is slow.
        X = graph.tolil()   # for better iteration

        # for details about ElasticNet, refer to https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
        alpha = 2 * self.config['l2_coef'] + self.config['l1_coef']
        l1_ratio = self.config['l1_coef'] / alpha
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            positive=True,
            fit_intercept=False,
            copy_X=False,
            precompute=True,
            selection='random',
            max_iter=self.config['epochs'],
            tol=1e-4
        )
        item_coeffs = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            iter_data = (
                tqdm(
                    range(X.shape[1]),
                    total=X.shape[1],
                    desc="Solving",
                    dynamic_ncols=True
                ) if verbose == 2 else range(X.shape[1])
            )
            for j in iter_data:
                r = X[:, j]
                X[:, j] = 0
                model.fit(X, r.A)
                item_coeffs.append(model.sparse_coef_)
                X[:, j] = r
        B = ssp.vstack(item_coeffs).T   # parameter
        self.item_similarity = B
        self.user_item = graph
        return
    