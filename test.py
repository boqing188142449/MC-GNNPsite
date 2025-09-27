import pandas as pd
import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from Model.GNN import CombinedGNN
from tqdm import tqdm
from Utils.GetDataset import GetDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, confusion_matrix

def Metric(preds, labels):
    """
    Calculate performance metrics for model predictions.

    Args:
        preds (np.array): Predicted probabilities
        labels (np.array): Ground truth labels

    Returns:
        tuple: (accuracy, recall, precision, f1, mcc, auc, auprc, specificity)
    """
    true_labels = np.array(labels).reshape(-1)
    pred_labels = np.array(preds).reshape(-1)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, np.round(pred_labels)).ravel()

    # Calculate various performance metrics
    recall = recall_score(true_labels, np.round(pred_labels))
    precision = precision_score(true_labels, np.round(pred_labels), zero_division=1)
    acc = accuracy_score(true_labels, np.round(pred_labels))
    f1 = f1_score(true_labels, np.round(pred_labels))
    mcc = matthews_corrcoef(true_labels, np.round(pred_labels))
    AUC = roc_auc_score(true_labels, pred_labels)
    AUPRC = average_precision_score(true_labels, pred_labels)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return acc, recall, precision, f1, mcc, AUC, AUPRC, specificity

# Load test dataset
test_root = '/media/2t/zhangzhi/Project01/Dataset/Protein/Test_72'  # Root directory containing protein data
test_dataset = GetDataset(test_root)
test_data_list = test_dataset.get_all()
test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

models = []
folds = 10
in_features = 2328
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
output_dir = '/media/2t/zhangzhi/Project01/output_pt/'  # Directory to save model files and results

# Load all fold models
for fold in range(folds):
    model_path = f'{output_dir}best_model_fold{fold + 1}.pt'
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found, skipping fold {fold + 1}")
        continue
    model = CombinedGNN(in_features=in_features, out_features=1, hidden_features=1024, dropout=0.2).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)

print('model count:', len(models))

if len(models) == 0:
    print("Error: No models loaded. Please check if model files exist in /media/2t/zhangzhi/Project01/output_pt/")
    exit(1)

# Store predictions for each fold
fold_preds = {i: [] for i in range(1, folds + 1)}
test_Y = []
with torch.no_grad():
    for data in tqdm(test_loader, desc="Evaluating"):
        data = data.to(device)
        y = data.y.view(-1, 1).float().to(device)
        test_Y.append(y.cpu().detach().numpy())
        for idx, model in enumerate(models, 1):
            outputs = model(data.x, data.edge_index1, data.edge_index2, data.edge_index3, data.edge_index4)
            fold_preds[idx].append(outputs.detach().cpu().numpy())

# Concatenate predictions and true labels
test_Y = np.concatenate(test_Y)
fold_metrics = {}
for fold in range(1, len(models) + 1):
    fold_preds[fold] = np.concatenate(fold_preds[fold])
    fold_metrics[fold] = Metric(fold_preds[fold], test_Y)

# Save all fold results to a text file
output_txt_path = f'{output_dir}72_results.txt'
with open(output_txt_path, 'w') as f:
    f.write('Test Results (Generated on 2025-09-25 00:26 PDT)\n')  # Updated to current time
    f.write('============================================\n\n')
    for fold in range(1, len(models) + 1):
        f.write(f'Fold {fold}:\n')
        f.write(f'ACC: {fold_metrics[fold][0]:.3f}\n')
        f.write(f'Recall: {fold_metrics[fold][1]:.3f}\n')
        f.write(f'Precision: {fold_metrics[fold][2]:.3f}\n')
        f.write(f'F1: {fold_metrics[fold][3]:.3f}\n')
        f.write(f'MCC: {fold_metrics[fold][4]:.3f}\n')
        f.write(f'AUC: {fold_metrics[fold][5]:.3f}\n')
        f.write(f'AUPRC: {fold_metrics[fold][6]:.3f}\n')
        f.write(f'Specificity: {fold_metrics[fold][7]:.3f}\n\n')
    # Compute and save average and standard deviation results
    mean_metrics = np.mean([list(m) for m in fold_metrics.values()], axis=0)
    std_metrics = np.std([list(m) for m in fold_metrics.values()], axis=0)
    f.write('Overall Average ± Standard Deviation:\n')
    f.write(f'ACC: {mean_metrics[0]:.3f} ± {std_metrics[0]:.3f}\n')
    f.write(f'Recall: {mean_metrics[1]:.3f} ± {std_metrics[1]:.3f}\n')
    f.write(f'Precision: {mean_metrics[2]:.3f} ± {std_metrics[2]:.3f}\n')
    f.write(f'F1: {mean_metrics[3]:.3f} ± {std_metrics[3]:.3f}\n')
    f.write(f'MCC: {mean_metrics[4]:.3f} ± {std_metrics[4]:.3f}\n')
    f.write(f'AUC: {mean_metrics[5]:.3f} ± {std_metrics[5]:.3f}\n')
    f.write(f'AUPRC: {mean_metrics[6]:.3f} ± {std_metrics[6]:.3f}\n')
    f.write(f'Specificity: {mean_metrics[7]:.3f} ± {std_metrics[7]:.3f}\n')

print(f"test_results saved to {output_txt_path}")

# Print average and standard deviation results
mean_metrics = np.mean([list(m) for m in fold_metrics.values()], axis=0)
std_metrics = np.std([list(m) for m in fold_metrics.values()], axis=0)
print(
    'test_ACC:%.3f±%.3f,test_Rec:%.3f±%.3f,test_Pre:%.3f±%.3f,test_F1:%.3f±%.3f,test_MCC:%.3f±%.3f,test_AUC:%.3f±%.3f,test_AUPRC:%.3f±%.3f,test_Spec:%.3f±%.3f' %
    (mean_metrics[0], std_metrics[0], mean_metrics[1], std_metrics[1], mean_metrics[2], std_metrics[2],
     mean_metrics[3], std_metrics[3], mean_metrics[4], std_metrics[4], mean_metrics[5], std_metrics[5],
     mean_metrics[6], std_metrics[6], mean_metrics[7], std_metrics[7]))