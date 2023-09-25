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
                for handler in self.logger.handlers:
                    handler.close()
                    self.logger.removeHandler(handler)
                print('Logger close successfully.')
            else:
                print('Logger is None. Do nothing.')
        except:
            print('Exception in closing logger')