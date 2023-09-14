       
import numpy as np
import numba
from numba.typed import Dict
from numba.core import types
from sklearn import metrics
from torch import topk
from unirec.facility.evaluation.evaluator_abc import *
from unirec.utils.general import get_topk_index

@numba.jit(nopython=True)
def _get_ndcg_weights(length):
    ndcg_weights = 1 / np.log2(np.arange(2, length + 2))
    return ndcg_weights

@numba.jit(nopython=True)
def _get_mrr_weights(length):
    mrr_weights = 1 / np.arange(1, length + 1)
    return mrr_weights

@numba.jit(nopython=True, parallel=True)
def get_rank(A):
    rank = np.empty(len(A), dtype=np.int32)
    for i in numba.prange(len(A)):
        a = A[i]
        key = a[0]
        r = 0
        for j in range(1, len(a)):
            if a[j] > key:
                r += 1
        rank[i] = r
    return rank


def _get_group_freq(topk_items: np.ndarray, item2group: np.ndarray, k: int):
    r"""Calculate the frequency of occurrence of each group of items in topk items.
    
    Args:
        topk_items (np.ndarray): topk item ids (sorted by scores in descending), shape [N, K] (N is batch size)
        item2group (np.ndarray): group id of items. 1-D array, i-th element represents the group id of item i.
        k (int): cutoff

    Return:
        np.ndarray: frequency of occurrence of each group. 1-D array with shape [#groups,]
    """
    n_groups = max(item2group) + 1
    res = np.zeros(n_groups)
    unique_itemid, counts = np.unique(topk_items[:, :k].reshape(-1), return_counts=True)
    for i, gid in enumerate(item2group[unique_itemid]):
        res[gid] += counts[i]
    return res


def cal_popkl_metric(group_freq: np.ndarray, item2group: np.ndarray, item2popularity: np.ndarray):
    r"""Calculate Pop-KL metrics for alignment objective.
    
    We consider aligning the frequency in prediction result to the popularity in training data.

    Args:
        group_freq (np.ndarray): frequency of each group in prediction. 1-D array, shape [#groups,]
        item2group (np.ndarray): group id of items. 1-D array, i-th element represents the group id of item i.
        item2popularity (np.ndarray): popularity of each item. 1-D array, i-th element represents the popularity of item i.
    
    Return:
        float: the value of Pop-KL
    """
    n_groups = max(item2group) + 1
    def get_group_freq(item_freq):
        group_freq = np.zeros(n_groups)
        for i in range(n_groups):
            _idx = item2group==i
            if len(_idx) > 0:
                group_freq[i] = item_freq[_idx].sum()
        return group_freq

    expect_dist = get_group_freq(item2popularity)
    expect_dist = expect_dist / expect_dist.sum()
    topk_dist = group_freq / group_freq.sum()
    return torch.nn.functional.kl_div(torch.from_numpy(topk_dist+1e-10).log(), 
                                      torch.from_numpy(expect_dist+1e-10).log(), 
                                      reduction='sum', log_target=True).numpy()


class OnePositiveEvaluator(Evaluator):    
    def __init__(self, metrics_str=None, group_size=-1, config=None, accelerator=None):
        super(OnePositiveEvaluator, self).__init__(metrics_str, group_size, config, accelerator)
        self.metrics_list = eval(metrics_str) 
        self.group_size = group_size
        self.noise = {}
        self.zero_vec = {}
        self.zero_vec_mask_k = {}
        
    def get_zero_vec(self, k):
        if k not in self.zero_vec:
            self.zero_vec[k] = np.zeros((k,), dtype=np.float32)
        return self.zero_vec[k]

    def get_zero_vec_mask_k(self, n, k):
        if (n,k) not in self.zero_vec_mask_k:
            if k == np.Inf:
                self.zero_vec_mask_k[(n,k)] = np.ones((n,), dtype=np.float32)  
            else:
                self.zero_vec_mask_k[(n,k)] = np.zeros((n,), dtype=np.float32)  
                self.zero_vec_mask_k[(n,k)][:k] = 1.0
            
        return self.zero_vec_mask_k[(n,k)]

    def ndcg(self, k, rank, w):    
        masker = self.get_zero_vec_mask_k(len(w), k)
        res = w[rank] * masker[rank] 
        return res
    
    def hit(self, k, rank):
        top_items = rank < k
        return top_items + 0.0 ## convert bool to float
    
    def evaluate_with_scores(self, scores, labels=None, prices=None, **kwargs):
        S = scores
        if self.group_size > 0:
            S = S.reshape(-1, self.group_size)  
            if prices is not None:
                prices = prices.reshape(-1, self.group_size)  
        
        if prices is not None and len(prices.shape)>0:
            prices = prices[:, 0] ## because onepos evaluation

        res = {}
        num_samples = S.shape[0]
        num_scores = S.shape[1] 
        
        # add small perturbation
        shape_key = S.shape
        if shape_key not in self.noise:
            self.noise[shape_key] = np.random.uniform(low=-1e-8, high=1e-8, size=S.shape)
        S += self.noise[shape_key]
        rank = get_rank(S) 

        if self._topk_flag:
            pos_itemids = kwargs['pos_itemids']
            # restore the score matrix
            S[np.arange(pos_itemids.shape[0]), pos_itemids] = S[:, 0]
            S[:, 0] = -np.Inf
            topk_itemids = get_topk_index(S, self._max_cutoff)
        
        ndcg_w = _get_ndcg_weights(num_scores)
        mrr_w = _get_mrr_weights(num_scores) 
        
        for metric in self.metrics_list:
            if 'group_auc' == metric:
                res[metric] = (num_scores - 1 - rank) / (num_scores - 1)
            elif 'auc' == metric:
                res[metric] = metrics.roc_auc_score(labels.reshape([-1, 1]), S.reshape([-1, 1]))
            elif 'ndcg' == metric:                
                res['ndcg'] = self.ndcg(np.Inf, rank, ndcg_w)
            elif 'mrr' == metric:                
                res['mrr'] = mrr_w[rank]
            elif 'least-misery' == metric:
                positem_gid = self.item2category[kwargs['pos_itemids']]
                res['_group_id'] = positem_gid
            elif '@' in metric:
                tokens = metric.split('@')
                key, ks = tokens[0], tokens[1].split(';')
                if key == 'ndcg':
                    for k in ks:
                        res['{0}@{1}'.format(key, k)] = self.ndcg(int(k), rank, ndcg_w)
                elif key == 'rndcg':
                    for k in ks:
                        res['{0}@{1}'.format(key, k)] = self.ndcg(int(k), rank, ndcg_w) * prices           
                elif key == 'hit':
                    for k in ks:
                        res['{0}@{1}'.format(key, k)] = self.hit(int(k), rank)  
                elif key == 'rhit':
                    for k in ks:
                        res['{0}@{1}'.format(key, k)] = self.hit(int(k), rank) * prices
                elif key == 'mrr':
                    for k in ks:
                        k = int(k)
                        masker = self.get_zero_vec_mask_k(len(mrr_w), k) 
                        res['{0}@{1}'.format(key, k)] =  masker[rank] * mrr_w[rank]
                elif key == 'recall' or key == 'rrecall':
                    pass ## no need to support recall for onspos evaluation. Because it equals to hit.
                elif key == 'pop-kl':
                    for k in ks:
                        k = int(k)
                        res['{0}@{1}'.format(key, k)] = _get_group_freq(topk_itemids, self.item2popularity_group, k)
                else:
                    raise ValueError('metric {0} is unknown.'.format(key))

        return res

    def merge_scores(self, all_results):
        overall_scores =  self.merge_scores_core(all_results)
        all_res = overall_scores
        return all_res

    def merge_scores_core(self, all_results):   
        if isinstance(all_results, list):
            if len(all_results) > 0:
                res = {}
                keys = all_results[0].keys()
                for metric_key in keys:
                    if metric_key.startswith("pop-kl"):
                        group_freq = np.stack([t[metric_key] for t in all_results]).sum(axis=0)
                        res[metric_key] = cal_popkl_metric(group_freq, self.item2popularity_group, self.item2popularity)
                    else:
                        res[metric_key] = np.concatenate([t[metric_key] for t in all_results])  
            else: 
                res = all_results[0]   
        else: 
            res = all_results
        
        keys = res.keys()
         
        all_res = {} 
        for key in keys:
            if not key.startswith('_'): # not temporary record
                all_res[key] = res[key].mean()     

        # least misery
        if 'least-misery' in self.metrics_list: 
            n_group = int(self.item2category.max()) + 1
            gid = res['_group_id']
            group_idx = []
            for i in range(1, n_group):
                group_idx.append(gid==i)

            for m in self.metrics_list:
                if (m != 'least-misery'):
                    if (res[m].shape==gid.shape):
                        all_res[f"min-{m}"] = min([res[m][idx].mean() for idx in group_idx])
                else:
                    pass
        return all_res
                    
 
if __name__ == '__main__':
    config={'verbose':2, 'data_format': 'user-item-label' }
    evaluator = OnePositiveEvaluator("['group_auc', 'hit@1;5', 'rhit@1;5', 'ndcg@1;5', 'ndcg', 'mrr', 'mrr@1;5']", config=config)
    data = np.random.uniform(low=-1e-8, high=1e-8, size=(100,20)) 
    prices = np.random.uniform(low=0, high=100, size=(100,20)) 
    pos_addon = np.random.uniform(low=0.3, high=0.7, size=(100, 1)) 
    neg_addon = np.random.uniform(low=0.0, high=0.4, size=(100, 19)) 
    data[:, :1] += pos_addon
    data[:, 1:] += neg_addon
    # labels = np.random.binomial(1, 0.5, data.shape).reshape(data.shape)
    labels = np.zeros_like(data)
    labels[:, :1] = 1
    session_ids = np.arange(len(data), dtype=np.int32)
    session_ids = np.repeat(session_ids, data.shape[1]).reshape(-1).tolist()
    res = evaluator.evaluate_with_scores(data, labels, prices) 
    b = evaluator.merge_scores(res)
    print(b)

    from sklearn.metrics import ndcg_score, roc_auc_score
    print('sklearn ndcg: {0}'.format(ndcg_score(labels, data)))
    print('sklearn ndcg@1: {0}'.format(ndcg_score(labels, data, k=1)))
    print('sklearn ndcg@5: {0}'.format(ndcg_score(labels, data, k=5)))

    print('sklearn AUC: {0}'.format(roc_auc_score(labels.reshape(-1), data.reshape(-1))))
    print('sklearn AUC: {0}'.format(np.mean(np.array([roc_auc_score(label, score) for label, score in zip(labels, data)]))))