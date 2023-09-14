import numpy as np
import scipy.sparse as ssp

from unirec.constants.protocols import *
from .seqrecdataset import SeqRecDataset

class AERecDataset(SeqRecDataset):
    r"""
    AERecDataset is designed for AutoEncoder models. The output in training and evaluation are different.
    In training mode, the output is (userid, item_seq), which contains the userid and users' history. 
    In evaluation mode, the output keeps the seem as SeqRecDataset, which adds user history with the key item_seq to each sample.
    """

    def _post_process(self):
        if self.config['data_loader_task'] == 'train':
            data_format = self.config['data_format']
            raw_data_pd = self.dataset_df
            if data_format in {DataFileFormat.T4.value}:
                self.logger.info(f'Explode the {ColNames.ITEMID_GROUP.value}.')
                _n_rows0 = len(raw_data_pd)
                raw_data_pd = raw_data_pd.explode([ColNames.ITEMID_GROUP.value, ColNames.LABEL_GROUP.value])
                raw_data_pd = raw_data_pd.rename(columns={ColNames.ITEMID_GROUP.value:ColNames.ITEMID.value, ColNames.LABEL_GROUP.value: ColNames.LABEL.value})
                _n_rows1 = len(raw_data_pd)
                self.logger.info('{0} / {1} rows in the new dataframe.'.format(_n_rows1, _n_rows0))
                data_format = DataFileFormat.T2.value

            if data_format in {DataFileFormat.T2.value, DataFileFormat.T2_1.value}:
                self.logger.info('Remove rows with label == 0')
                _n_rows0 = len(raw_data_pd)
                raw_data_pd = raw_data_pd[raw_data_pd[ColNames.LABEL.value] > 0]
                _n_rows1 = len(raw_data_pd)
                self.logger.info('{0} / {1} rows remains.'.format(_n_rows1, _n_rows0))
            
            if data_format in {DataFileFormat.T1.value, DataFileFormat.T2.value, DataFileFormat.T2_1.value, DataFileFormat.T3.value}:
                self.logger.info('Group interactions by user_id.')
                _n_rows0 = len(raw_data_pd)
                raw_data_pd = raw_data_pd.groupby(ColNames.USERID.value)[ColNames.ITEMID.value].apply(lambda x:np.array(x)).to_frame().reset_index()
                raw_data_pd = raw_data_pd.rename(columns={ColNames.ITEMID.value: ColNames.USER_HISTORY.value})
                _n_rows1 = len(raw_data_pd)
                self.logger.info('{0} / {1} rows after group'.format(_n_rows1, _n_rows0))
            elif data_format in {DataFileFormat.T5.value, DataFileFormat.T6.value}:
                pass
            else:
                raise NotImplementedError()
            #
            # Remove duplicated interactions for each user (maybe those items are happened in different time). 
            self.logger.info('Remove duplicated interactions in user history.')
            _n_inter_before_remove_dup = np.sum([len(x) for x in raw_data_pd[ColNames.USER_HISTORY.value]])
            raw_data_pd[ColNames.USER_HISTORY.value] = raw_data_pd[ColNames.USER_HISTORY.value].apply(lambda x: np.unique(x))
            _n_inter_after_remove_dup = np.sum([len(x) for x in raw_data_pd[ColNames.USER_HISTORY.value]])
            self.logger.info('{0} / {1} interactions remain'.format(_n_inter_after_remove_dup, _n_inter_before_remove_dup))

            self.dataset_df = raw_data_pd[[ColNames.USERID.value, ColNames.USER_HISTORY.value]]
        else:
            pass


    def set_return_column_index(self):
        if self.config['data_loader_task'] == 'train':  # for train file
            self.return_key_2_index = {
                ColNames.USERID.value: 0,
                ColNames.USER_HISTORY.value: 1,
            }
            if self.use_features:
                self.return_key_2_index['item_seq_features'] = len(self.return_key_2_index)
        else:   # for test file
            super(AERecDataset, self).set_return_column_index()
        

    def __getitem__(self, index):
        if self.config['data_loader_task'] == 'train':
            sample = self.dataset[index]   
            elements = (sample[0], sample[1])  # train: [user_id, item_seq];
            if self.use_features:
                item_seq_features = self.item2features[sample[1]]
                elements = elements + (item_seq_features,)
            return elements   
        else:
            return super(AERecDataset, self).__getitem__(index)


    def get_graph(self, format='csr'):
        r""" Return a sparse graph of user-item interactions.
        Args:
            format(str): the sparse matrix format of the graph. Only csr-matrix
                is supported now. Default: 'csr'.
        Returns:
            sparse-matrix: sparse matrix of the user-item interaction graph. The shape
                is (n_user, u_item) 
        """
        if self.config['data_loader_task'] == 'train':
            if format == 'csr':
                n_u, n_i = self.config['n_users'], self.config['n_items']
                num_inter = np.zeros((n_u,))
                indptr = np.zeros((n_u+1, ))
                #
                # To build the user-item graph, whose i-th row indicates the user i, 
                # we need to sort the dataset with the user_id first and then we can 
                # concatenate the history of all users directly as the indices of the 
                # csr-matrix. And the indptr could be obtained by cumsum of the length 
                # of users sorted by id.
                sorted_dataset = self.dataset[self.dataset[:, 0].argsort()]
                indices = np.concatenate(sorted_dataset[:, 1], axis=0)
                data = np.ones_like(indices, dtype=np.float64)
                for s in self.dataset:
                    u, h = s[0], s[1]
                    num_inter[u] = len(h)
                indptr[1:] = np.cumsum(num_inter)
                ui_graph = ssp.csr_matrix((data, indices, indptr), shape=(n_u, n_i))
                self.logger.debug("user-item graph are constructed.")
            else:
                raise NotImplementedError("other formats of graph are not supported now.")
            return ui_graph
