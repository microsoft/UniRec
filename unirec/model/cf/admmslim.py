import numpy as np
from tqdm import tqdm

from .ease import EASE


def soft_threshold(x, threshold):
    r"""Calculating soft threshold for x. The formula is described in code."""
    return (np.abs(x) > threshold) * (np.abs(x) - threshold) * np.sign(x)

class AdmmSLIM(EASE):
    r""" AdmmSLIM proposed a faster optimization method for SLIM model.
    For more details, please refer to https://dl.acm.org/doi/10.1145/3336191.3371774.
    """
    # Note: the item_similarity parameters in AdmmSLIM is a dense matrix of shape (num_item, num_item),
    # and the solving procedure will use another two matrices with the same shape,
    # which may cause out-of-memory issue in solving. 
    # For example, a dataset with 100,000 items will occupy more than 100GB memory
    # For dataset with about 10,000 items, 10GB memory is occupied.
    def solve(self, graph, verbose=2):
        """ Optimize the parameters.
        Args:
            graph(scipy.sparse.spmatrix): the graph of user-item interaction graph, which is a sparse matrix.
            verbose (int): Control wether to show the progress of training epoch and evaluate epoch. 
                            0: show nothing;  1: show basic progress message, but no tqdm progress bar; 2: show everything
        """
        rho = self.config['admm_penalty']
        lambd = [self.config['l1_coef'], self.config['l2_coef'] * 2]
        alpha = self.config['item_spec_reg']
        X = graph
        item_means = X.mean(axis=0).A
        XtX = (X.T @ X).A
        delta = lambd[1] * np.diag(np.power(item_means, alpha)) + rho * np.identity(X.shape[1])
        P = np.linalg.inv(XtX + delta).astype(np.float32)
        B_aux = (P @ XtX).astype(np.float32)
        Gamma = np.zeros(XtX.shape, dtype=np.float32)
        C = np.zeros(XtX.shape, dtype=np.float32)

        # iterate util convergence
        iter_data = (
            tqdm(
                range(self.config['epochs']),
                total=self.config['epochs'],
                desc="Solving",
                dynamic_ncols=True
            ) if verbose == 2 else range(self.config['epochs'])
        )
        for _ in iter_data:
            B_tilde = B_aux + P @ (rho * C - Gamma)
            gamma = np.diag(B_tilde) / (np.diag(P) + 1e-7)
            B = B_tilde - P * gamma
            C = soft_threshold(B + Gamma / rho, lambd[0] / rho)
            C = (C > 0) * C
            Gamma += rho * (B - C)

        self.item_similarity = C
        self.user_item = X
        return
