import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

from utils import DEVICE, log_epoch, calculate_multiclass_metrics, plot_confusion_matrix, split_data, plot_training_history

class EarlyStopping:
    """Останавливает обучение, если потери на валидации не улучшаются."""
    def __init__(self, patience=7, verbose=False, delta=0, path=os.path.join(MODELS_DIR, 'checkpoint.pt')):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'Счетчик EarlyStopping: {self.counter} из {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):#Сохраняет модель при уменьшении потерь на валидации.
        if self.verbose:
            logging.info(f'({self.val_loss_min:.6f} --> {val_loss:.6f}). Сохраняем модель')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

def train_linear_regression_with_modifications(X_train, y_train, X_val, y_val, in_features):
    logging.info("Обучение модели с регуляризацией и остановкой")
    device = DEVICE
    model = LinearRegression(in_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # небольшие коэффициенты регуляризации
    l1_lambda, l2_lambda = 1e-4, 1e-4

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    epochs = 200
    checkpoint_path = os.path.join(MODELS_DIR, 'linear_reg_best.pt')
    early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint_path)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            mse_loss = criterion(outputs, batch_y)
            l1_reg = sum(p.abs().sum() for p in model.parameters())
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = mse_loss + l1_lambda * l1_reg + l2_lambda * l2_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        log_epoch(epoch, epochs, avg_train_loss, {'val_loss': avg_val_loss})

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logging.info("Остановка")
            break
            
    model.load_state_dict(torch.load(checkpoint_path))
    logging.info(f"Лучшая версия модели загружена из {checkpoint_path}.")
    return model, history

class SoftmaxRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

def train_softmax_regression(X_train, y_train, X_val, y_val, in_features, num_classes):
    logging.info("Обучение Softmax-регрессии")
    device = DEVICE
    model = SoftmaxRegression(in_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    epochs = 100
    history = {'train_loss': [], 'val_loss': [], 'val_f1_score': [], 'val_roc_auc': []}
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss, all_val_preds_labels, all_val_preds_probs, all_val_true = 0, [], [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_val_preds_labels.extend(preds.cpu().numpy())
                all_val_preds_probs.extend(probs.cpu().numpy())
                all_val_true.extend(batch_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics = calculate_multiclass_metrics(np.array(all_val_true), np.array(all_val_preds_probs), np.array(all_val_preds_labels))
        history['val_loss'].append(avg_val_loss)
        history['val_f1_score'].append(val_metrics['f1_score'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        log_epoch(epoch, epochs, avg_train_loss, {'val_loss': avg_val_loss, **val_metrics})

    return model, history, all_val_true, all_val_preds_labels

if __name__ == '__main__':
    from homework_datasets import CSVDataset

    data_dir = os.path.join(BASE_DIR, 'data')
    titanic_csv = os.path.join(data_dir, 'Titanic.csv')
    spotify_csv = os.path.join(data_dir, 'spotify.csv')

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Бинарная классификация Survived
    if os.path.exists(titanic_csv):
        ds = CSVDataset(titanic_csv, target_col='Survived', task='classification', sample_frac=0.3)
        X_tr, X_val, _, y_tr, y_val, _ = split_data(ds.X, ds.y)
        clf_model, clf_hist, y_true, y_pred = train_softmax_regression(
            X_tr, y_tr, X_val, y_val,
            in_features=ds.X.shape[1], num_classes=ds.num_classes
        )
        plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(PLOTS_DIR, 'titanic_cm.png'))

    # Регрессия популярности песен
    if os.path.exists(spotify_csv):
        ds_r = CSVDataset(spotify_csv, target_col='popularity', task='regression', sample_frac=0.1)
        X_tr, X_val, _, y_tr, y_val, _ = split_data(ds_r.X, ds_r.y)
        reg_model, reg_hist = train_linear_regression_with_modifications(
            X_tr, y_tr, X_val, y_val, in_features=ds_r.X.shape[1]
        )
        plot_training_history(reg_hist, save_path=os.path.join(PLOTS_DIR, 'spotify_reg.png'))