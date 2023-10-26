# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import time
import random

from unirec.utils.general import *

class Logger(object):  
    def __init__(self, log_dir, name=None, time_str='', rand='', is_main_process=True):  
        self.CRITICAL = logging.CRITICAL
        self.FATAL = logging.FATAL
        self.ERROR = logging.ERROR
        self.WARNING = logging.WARNING 
        self.INFO = logging.INFO
        self.DEBUG = logging.DEBUG 
        
        self.logger_time_str = time_str 
        self.logger_rand = rand 

        if name is None:
            name = __name__
            
        self.is_main_process = is_main_process
        if not self.is_main_process:
            self.logger = logging.getLogger(name) 
            self.logger.propagate = False
        else:
            os.makedirs(log_dir, exist_ok=True) 
            filename = os.path.join(
                log_dir, 
                '{0}.{1}.{2}.txt'.format(
                    name, 
                    self.logger_time_str,
                    self.logger_rand
                )
            )

            print('Writing logs to ' + filename)
            self.filename = filename
            self.logger = logging.getLogger(name) # 
            self.logger.setLevel(logging.DEBUG)
            
            # remove all potential handlers in case that handlers not closed before, especially in notebook environment
            if len(self.logger.handlers) > 0:
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)

            handler = logging.FileHandler(filename)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            handler02 = logging.StreamHandler()
            handler02.setLevel(logging.DEBUG)
            handler02.setFormatter(formatter)
            self.logger.addHandler(handler02)
    
    def log(self, level, message):
        self.logger.log(level, message)    
    
    def remove_handles(self):
        try:
            if self.logger is not None:
                for handler in self.logger.handlers[:]:
                    # Note: here [:] operator create a shallow copy of the handler list, avoiding the case that
                    # the remove/delete operation would cause some elements in list are not traversed
                    # For example, if the list a is [0,1], then we use a for-loop to delete all elements. If we use the
                    # following code:
                    # >>> for i in a:
                    # >>>     a.remove(i)
                    # >>> print(a)
                    # We will get a=[1] instead of [], which is not expected.
                    # In the logger case, the handlers would not be closed completely without the operator [:].
                    handler.close()
                    self.logger.removeHandler(handler)
                print('Logger close successfully.')
            else:
                print('Logger is None. Do nothing.')
        except:
            print('Exception in closing logger')