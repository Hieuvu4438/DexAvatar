"""
Vendored from SOKE (https://github.com/2000ZRL/SOKE).
Original: mGPT/utils/misc.py (neq_load_customized)
"""
import torch


def neq_load_customized(model, pretrained_dict, verbose=False):
    """Load pre-trained model with shape-mismatch tolerance."""
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            tmp[k] = v
        else:
            if verbose:
                print(k)
    if verbose:
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
            elif model_dict[k].shape != pretrained_dict[k].shape:
                print(k, 'shape mis-matched, not loaded')
        print('===================================\n')
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model
