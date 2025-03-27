

def change_classifier(model, classifier):
    ## Change the classifier
    if model.__class__.__name__ == 'ShuffleNetV2':
        classifier[-1].in_features = model.fc.in_features
        model.fc = classifier
    else:
        classifier[-1].in_features = model.classifier[-1].in_features
        model.classifier = classifier
        
    return model