from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score, hamming_loss, accuracy_score
import torch
from torchmetrics.classification import MultilabelAccuracy
import numpy as np
import pandas as pd
import mlflow

# TN FP
# FN TP



def create_zero_metrics():
    zero_metrics = {
        'averaged_example_based_accuracy': 0.0,
        'f1_score': {
            'macro_averaged': 0.0,
            'micro_averaged': 0.0,
            'per_label': np.zeros(40, dtype=float),  # Assuming the array has 40 elements
            'sample_average': 0.0,
            'weighted_averaged': 0.0
        },
        'hamming_loss': 0.0,
        'precision_score': {
            'macro_averaged': 0.0,
            'micro_averaged': 0.0,
            'per_label': np.zeros(40, dtype=float),
            'sample_average': 0.0,
            'weighted_averaged': 0.0
        },
        'recall_score': {
            'macro_averaged': 0.0,
            'micro_averaged': 0.0,
            'per_label': np.zeros(40, dtype=float),
            'sample_average': 0.0,
            'weighted_averaged': 0.0
        }
    }
    return zero_metrics



def calculate_prediction_results_per_label(y_true, y_pred):
    MCM = multilabel_confusion_matrix(y_true, y_pred)

    preds = {
        'true_positives' : MCM[:, 1, 1],
        'false_positives' : MCM[:, 1, 0],
        'true_negatives': MCM[:, 0, 0],
        'false_negatives': MCM[:, 1, 0]
    }
    return preds




def calculate_prediction_results_per_sample(y_true, y_pred):
    out = multilabel_confusion_matrix(y_true, y_pred, samplewise=True)
    print('Actual: ', y_true)
    print('Predicted: ', y_pred)
    return out



def calculate_precision_scores(y_true, y_pred):
    precison_scores = {
        'per_label': precision_score(y_true, y_pred, average=None, zero_division=0.0),
        'micro_averaged': precision_score(y_true, y_pred, average='micro', zero_division=0.0),
        'macro_averaged': precision_score(y_true, y_pred, average='macro', zero_division=0.0),
        'weighted_averaged': precision_score(y_true, y_pred, average='weighted', zero_division=0.0),
        'sample_average': precision_score(y_true, y_pred, average='samples', zero_division=0.0)
    }
    return precison_scores





def calculate_recall_scores(y_true, y_pred):
    recall_scores = {
        'per_label': recall_score(y_true, y_pred, average=None, zero_division=0.0),
        'micro_averaged': recall_score(y_true, y_pred, average='micro', zero_division=0.0),
        'macro_averaged': recall_score(y_true, y_pred, average='macro', zero_division=0.0),
        'weighted_averaged': recall_score(y_true, y_pred, average='weighted', zero_division=0.0),
        'sample_average': recall_score(y_true, y_pred, average='samples', zero_division=0.0)
    }
    return recall_scores




def calculate_f1_scores(y_true, y_pred): 
    f1_scores = {
        'per_label': f1_score(y_true, y_pred, average=None, zero_division=0.0),
        'micro_averaged': f1_score(y_true, y_pred, average='micro', zero_division=0.0),
        'macro_averaged': f1_score(y_true, y_pred, average='macro', zero_division=0.0),
        'weighted_averaged': f1_score(y_true, y_pred, average='weighted', zero_division=0.0),
        'sample_average': f1_score(y_true, y_pred, average='samples', zero_division=0.0)
    }
    return f1_scores




def evaluate_performance(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        threshold: int = 0.5
):
    # Create a binary representation of the predictions
    y_pred = (torch.nn.Sigmoid()(y_pred) > threshold).float()

    accuracy = MultilabelAccuracy(num_labels=y_pred.shape[1], threshold=threshold)


    evaluation_metrics = {
        'precision_score': calculate_precision_scores(y_true.numpy(), y_pred.numpy()),
        'recall_score': calculate_recall_scores(y_true.numpy(), y_pred.numpy()),
        'f1_score': calculate_f1_scores(y_true.numpy(), y_pred.numpy()),
        'hamming_loss': hamming_loss(y_true.numpy(), y_pred.numpy()),
        'averaged_example_based_accuracy': accuracy(y_pred, y_true).item(),
    }

    return evaluation_metrics



class MetricsLogger():
    def __init__(self, columns):
        self.df_f1 = pd.DataFrame(columns=columns)
        self.df_rec = pd.DataFrame(columns=columns)
        self.df_prec = pd.DataFrame(columns=columns)
        
    
    def log_metrics(self, stage, metrics, step):
        
        mlflow.log_metric(f"Loss_{stage}", f"{metrics['loss']:4f}", step=step)
        mlflow.log_metric(f"Hamming_loss_{stage}", f"{metrics['hamming_loss']:4f}", step=step)
        mlflow.log_metric(f'Subset_Accuracy_{stage}', f"{metrics['averaged_example_based_accuracy']:4f}", step=step)

        mlflow.log_metric(f'F1_Score_Micro_{stage}', f"{metrics['f1_score']['micro_averaged']:4f}", step=step)
        mlflow.log_metric(f'F1_Score_Macro_{stage}', f"{metrics['f1_score']['micro_averaged']:4f}", step=step)
        mlflow.log_metric(f'F1_Score_Sample_{stage}', f"{metrics['f1_score']['sample_average']:4f}", step=step)
        mlflow.log_metric(f'F1_Score_Weighted_{stage}', f"{metrics['f1_score']['weighted_averaged']:4f}", step=step)

        mlflow.log_metric(f'Recall_Micro_{stage}', f"{metrics['recall_score']['micro_averaged']:4f}", step=step)
        mlflow.log_metric(f'Recall_Macro_{stage}', f"{metrics['recall_score']['macro_averaged']:4f}", step=step)
        mlflow.log_metric(f'Recall_Sample_{stage}', f"{metrics['recall_score']['sample_average']:4f}", step=step)
        mlflow.log_metric(f'Recall_Weighted_{stage}', f"{metrics['recall_score']['weighted_averaged']:4f}", step=step)

        mlflow.log_metric(f'Precision_Micro_{stage}', f"{metrics['precision_score']['micro_averaged']:4f}", step=step)
        mlflow.log_metric(f'Precision_Macro_{stage}', f"{metrics['precision_score']['macro_averaged']:4f}", step=step)
        mlflow.log_metric(f'Precision_Sample_{stage}', f"{metrics['precision_score']['sample_average']:4f}", step=step)
        mlflow.log_metric(f'Precision_Weighted_{stage}', f"{metrics['precision_score']['weighted_averaged']:4f}", step=step)


        if stage == 'TEST':
            self.df_f1.loc[step] = metrics['f1_score']['per_label']
            self.df_rec.loc[step] = metrics['recall_score']['per_label']
            self.df_prec.loc[step] = metrics['precision_score']['per_label']
        
        
    
    def save_artifact(self):
        mlflow.log_table(self.df_f1, 'f1_score_per_label.json')
        mlflow.log_table(self.df_rec, 'recall_per_label.json')
        mlflow.log_table(self.df_prec, 'precision_per_label.json')