import numpy as np
from typing import Mapping



def add_dicts(dict1, dict2):
    """Recursively adds two dictionaries with the same structure."""
    if not dict1 or not dict2:
        return dict1 or dict2  # Return whichever is not empty

    result = {}
    for key in dict1:
        if isinstance(dict1[key], Mapping):  # If it's a nested dictionary, recurse
            result[key] = add_dicts(dict1[key], dict2[key])
        elif isinstance(dict1[key], np.ndarray):  # Element-wise sum for NumPy arrays
            result[key] = dict1[key] + dict2[key]
        else:  # Normal scalar sum
            result[key] = dict1[key] + dict2[key]

    return result




def divide_dict(d, divisor):
    """Recursively divides all elements of a dictionary by a divisor."""
    if divisor == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    
    if isinstance(d, dict):
        return {k: divide_dict(v, divisor) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return d / divisor
    elif isinstance(d, (int, float)):
        return d / divisor
    else:
        return d



def change_classifier(model, classifier):
    ## Change the classifier
    if model.__class__.__name__ == 'ShuffleNetV2':
        classifier[-1].in_features = model.fc.in_features
        model.fc = classifier
    else:
        classifier[-1].in_features = model.classifier[-1].in_features
        model.classifier = classifier
        
    return model



def freeze_model(model):
    ## Freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False

    ## Unfreeze the last layer
    if model.__class__.__name__ == 'ShuffleNetV2':
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model