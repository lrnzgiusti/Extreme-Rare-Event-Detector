# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:18:29 2019

@author: logiusti
"""


import time

def timeit(method):
    """
    Decorator for timing the execution speed of functions
    :param method: function to be timed.
    :return: decorated function
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print((method.__name__, round((te - ts),2)), "Args:", args[1:])
        return result

    return timed
