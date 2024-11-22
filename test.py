import os
import torch
import numpy as np
from torch_geometric.data import DataLoader
from Model.GNN import CombinedGNN

from tqdm import tqdm
from Utils.get_dataset import GetDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, confusion_matrix


def Metric(preds, labels):
    true_labels = np.array(labels).reshape(-1)
    pred_labels = np.array(preds).reshape(-1)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, np.round(pred_labels)).ravel()

    # Calculate metrics
    recall = recall_score(true_labels, np.round(pred_labels))  # Recall
    precision = precision_score(true_labels, np.round(pred_labels), zero_division=1)  # Precision
    acc = accuracy_score(true_labels, np.round(pred_labels))  # Accuracy
    f1 = f1_score(true_labels, np.round(pred_labels))  # F1-score
    mcc = matthews_corrcoef(true_labels, np.round(pred_labels))  # Matthews correlation coefficient
    AUC = roc_auc_score(true_labels, pred_labels)  # Area under ROC curve
    AUPRC = average_precision_score(true_labels, pred_labels)  # Area under precision-recall curve

    # Calculate specificity
    specificity = tn / (tn + fp)

    return acc, recall, precision, f1, mcc, AUC, AUPRC, specificity


# Load test data
test_root = '/media/2t/zhangzhi/Project01/dataset'  # Root directory of the dataset, containing protein folders
test_dataset = GetDataset(test_root)
test_data_list = test_dataset.get_all()
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

models = []
folds = 10
in_features = 2328
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for fold in range(folds):
    model_path = f'./output_pt/best_model_fold{fold + 1}.pt'
    if not os.path.exists(model_path):
        continue

    model = CombinedGNN(in_features=in_features, out_features=1, hidden_features=1024, dropout=0.2).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)

print('model count:', len(models))

test_preds = []
test_Y = []
with torch.no_grad():
    for data in tqdm(test_loader):
        data = data.to(device)
        outputs = [model(data.x, data.edge_index1, data.edge_index2, data.edge_index3, data.edge_index4) for model in
                   models]
        y = data.y.view(-1, 1).float().to(device)

        # Average predictions from the 5 models
        outputs = torch.stack(outputs, 0).mean(0)
        test_preds.append(outputs.detach().cpu().numpy())
        test_Y.append(y.cpu().detach().numpy())

# Concatenate predictions and true labels
test_preds = np.concatenate(test_preds)
test_Y = np.concatenate(test_Y)
test_metric = Metric(test_preds, test_Y)
print(
    'test_ACC:%.3f, test_Rec:%.3f, test_Pre:%.3f, test_F1:%.3f, test_MCC:%.3f, test_AUC:%.3f, test_AUPRC:%.3f, test_Spec:%.3f' %
    (test_metric[0], test_metric[1], test_metric[2], test_metric[3], test_metric[4], test_metric[5], test_metric[6],
     test_metric[7]))
