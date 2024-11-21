import os
import torch
import numpy as np
from torch_geometric.data import DataLoader
from Model.GNN import CombinedGNN
from tqdm import tqdm
from Utils.get_dataset import GetDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, confusion_matrix


# 计算各项指标的均值和标准差
def calculate_metrics_with_std(preds, labels, folds=10):
    true_labels = np.array(labels).reshape(-1)
    pred_labels = np.array(preds).reshape(-1)

    metrics_per_fold = []

    for fold in range(folds):
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(true_labels, np.round(pred_labels)).ravel()

        # 计算各项指标
        recall = recall_score(true_labels, np.round(pred_labels))  # Recall
        precision = precision_score(true_labels, np.round(pred_labels), zero_division=1)  # Precision
        acc = accuracy_score(true_labels, np.round(pred_labels))  # Accuracy
        f1 = f1_score(true_labels, np.round(pred_labels))  # F1-score
        mcc = matthews_corrcoef(true_labels, np.round(pred_labels))  # Matthews correlation coefficient
        AUC = roc_auc_score(true_labels, pred_labels)  # Area under ROC curve
        AUPRC = average_precision_score(true_labels, pred_labels)  # Area under precision-recall curve

        # 计算特异性
        specificity = tn / (tn + fp)

        # 保存每一折的结果
        metrics_per_fold.append([acc, recall, precision, f1, mcc, AUC, AUPRC, specificity])

    # 转换为numpy数组以便计算标准差
    metrics_per_fold = np.array(metrics_per_fold)

    # 计算每项指标的均值和标准差
    mean_metrics = np.mean(metrics_per_fold, axis=0)
    std_metrics = np.std(metrics_per_fold, axis=0)

    return mean_metrics, std_metrics


# 加载数据集
test_root = '/media/2t/zhangzhi/Project01/Dataset/Protein/Test_72'  # 数据集根目录，包含蛋白质文件夹
test_dataset = GetDataset(test_root)
test_data_list = test_dataset.get_all()
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

models = []
folds = 10
in_features = 2328
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载所有折的模型
for fold in range(folds):
    model_path = f'./output_pt/best_model_fold{fold + 1}.pt'
    if not os.path.exists(model_path):
        continue

    model = CombinedGNN(in_features=in_features, out_features=1, hidden_features=1024, dropout=0.2).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)

print('Model count:', len(models))

# 保存预测结果和真实标签
test_preds = []
test_Y = []

# 进行预测
with torch.no_grad():
    for data in tqdm(test_loader):
        data = data.to(device)
        # 每个模型的预测
        outputs = [model(data.x, data.edge_index1, data.edge_index2, data.edge_index3, data.edge_index4) for model in
                   models]
        y = data.y.view(-1, 1).float().to(device)

        # 对五个模型的预测取平均
        outputs = torch.stack(outputs, 0).mean(0)
        test_preds.append(outputs.detach().cpu().numpy())
        test_Y.append(y.cpu().detach().numpy())

# 将预测和真实标签合并
test_preds = np.concatenate(test_preds)
test_Y = np.concatenate(test_Y)

# 计算各项指标的均值和标准差
mean_metrics, std_metrics = calculate_metrics_with_std(test_preds, test_Y)

# 打印结果
print(
    'test_ACC:%.4f ± %.4f, test_Rec:%.4f ± %.4f, test_Pre:%.4f ± %.4f, test_F1:%.4f ± %.4f, test_MCC:%.4f ± %.4f, '
    'test_AUC:%.4f ± %.4f, test_AUPRC:%.4f ± %.4f, test_Spec:%.4f ± %.4f' %
    tuple(np.concatenate((mean_metrics, std_metrics))))
