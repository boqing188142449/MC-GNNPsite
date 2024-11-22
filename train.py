import random
from torch.utils.data import Subset
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from Utils.GetDataset import GetDataset
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from Model.GNN import CombinedGNN

def Seed_everything(seed=42):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def Metric(preds, labels):
    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)
    mcc = matthews_corrcoef(labels, np.round(preds))
    AUC = roc_auc_score(labels, preds)
    AUPRC = average_precision_score(labels, preds)
    return AUC, AUPRC, mcc

def train_epoch(model, optimizer, train_loader, criterion, device, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    train_preds = []
    train_Y = []
    model.train()
    bar = tqdm(train_loader)
    for batch_idx, data in enumerate(bar):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index1, data.edge_index2, data.edge_index3, data.edge_index4)
        y = data.y.view(-1, 1).float().to(device)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_preds.append(outputs.detach().cpu().numpy())
        train_Y.append(y.clone().detach().cpu().numpy())
        bar.set_description('loss: %.4f' % (loss.item()))
    train_preds = np.concatenate(train_preds)
    train_Y = np.concatenate(train_Y)
    metrics = Metric(train_preds, train_Y)
    print("Epoch {}: train_auc: {:.6f}, train_auprc: {:.6f} , MCC: {:.6f}\n".format(epoch + 1, metrics[0], metrics[1],
                                                                                    metrics[2]))

def validate(model, val_loader, device):
    print('Validating on {} samples...'.format(len(val_loader.dataset)))
    model.eval()
    valid_preds = []
    valid_Y = []
    for data in tqdm(val_loader):
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data.x, data.edge_index1, data.edge_index2, data.edge_index3, data.edge_index4)
        y = data.y.view(-1, 1).float().to(device)
        valid_preds.append(outputs.detach().cpu().numpy())
        valid_Y.append(y.clone().detach().cpu().numpy())
    valid_preds = np.concatenate(valid_preds)
    valid_Y = np.concatenate(valid_Y)
    valid_metric = Metric(valid_preds, valid_Y)
    return valid_metric

# Set data path
root = r'/media/2t/zhangzhi/Project01/Dataset/Protein/Train_DBSRP'
dataset = GetDataset(root)
data_list = dataset.get_all()

# Aggregate protein labels as new labels
labels = np.array([data.y.numpy().sum() for data in data_list])

# Ensure dataset and label counts match
assert len(data_list) == len(labels), "Dataset and label counts do not match"

# Hyperparameters
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
LR = 0.0006
LOG_INTERVAL = 5
NUM_EPOCHS = 500
N_SPLITS = 10
in_features = 2328

seed = 42
Seed_everything(seed=seed)
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

all_val_auc = []
all_val_auprc = []
# 10-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(data_list, labels)):
    print("\n========== Fold " + str(fold + 1) + " ==========")

    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn,
                            drop_last=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = CombinedGNN(in_features=in_features, out_features=1, hidden_features=1024, dropout=0.3).to(device)

    num_positive_samples = sum(torch.sum(data.y == 1).item() for data in train_dataset)
    num_negative_samples = sum(torch.sum(data.y == 0).item() for data in train_dataset)
    pos_weight = torch.tensor([num_negative_samples / num_positive_samples], device=device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Positive samples: {}, Negative samples: {}".format(num_positive_samples, num_negative_samples))
    print("Pos weight:", pos_weight.item())

    best_epoch = 0
    val_auc = 0
    best_auprc = 0
    not_improve_epochs = 0
    stop_count = 20

    # Training and validation
    for epoch in range(NUM_EPOCHS):
        train_epoch(model, optimizer, train_loader, criterion, device, epoch + 1)
        AUC, AUPRC, MCC = validate(model, val_loader, device)
        print("Epoch {}:  AUC: {:.3f}, AUPRC: {:.3f}, MCC: {:.3f}\n".format(epoch + 1, AUC, AUPRC, MCC))

        if AUPRC >= best_auprc:
            not_improve_epochs = 0
            val_auc = AUC
            best_auprc = AUPRC
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'./output_pt/best_model_fold{fold + 1}.pt')
            print('Improved AUPRC at epoch {}, best AUPRC: {:.3f}, AUC: {:.3f}\n'.format(best_epoch, AUPRC, AUC))
        else:
            not_improve_epochs += 1
            print('Epoch {}: AUPRC did not improve, best AUPRC: {:.3f}, AUC: {:.3f}\n'.format(best_epoch, best_auprc,
                                                                                              val_auc))
            if not_improve_epochs >= stop_count:
                print("Early stopping at epoch {}".format(epoch + 1))
                break

    state_dict = torch.load(f'./output_pt/best_model_fold{fold + 1}.pt', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    valid_preds = []
    valid_Y = []

    for data in tqdm(val_loader):
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data.x, data.edge_index1, data.edge_index2, data.edge_index3, data.edge_index4)
        valid_preds.append(outputs.cpu().numpy())
        valid_Y.append(data.y.cpu().numpy())

    valid_preds = np.concatenate(valid_preds)
    valid_Y = np.concatenate(valid_Y)
    valid_metric = Metric(valid_preds, valid_Y)
    all_val_auc.append(valid_metric[0])
    all_val_auprc.append(valid_metric[1])

# Output cross-validation results
print('Cross-validation results:')
print('AUC:', np.mean(all_val_auc), '±', np.std(all_val_auc))
print('AUPRC:', np.mean(all_val_auprc), '±', np.std(all_val_auprc))
