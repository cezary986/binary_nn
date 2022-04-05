import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/binary_nn')

def disable_sklearn_warnings():
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn