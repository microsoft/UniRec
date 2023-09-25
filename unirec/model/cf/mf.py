# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unirec.model.base.recommender import BaseRecommender

class MF(BaseRecommender):    
    def __init__(self, config):
        super(MF, self).__init__(config)
        ## every thing should be ready from the BaseRecommender class


