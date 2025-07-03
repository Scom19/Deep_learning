import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import math
import numpy as np

from utils import DEVICE, split_data, log_epoch
from homework_model_modification import LinearRegression, SoftmaxRegression
from homework_datasets import CSVDataset

# Основные константы скрипта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _single_train_run(
        X_tr,
        y_tr,
        X_val,
        y_val,
        n_in,
        opt_name: str = "sgd",
        lr: float = 1e-2,
        bs: int = 32,
):
    """Один цикл обучения; возвращает финальные потери на валидации"""
    device = DEVICE
    model = LinearRegression(n_in).to(device)
    criterion = nn.MSELoss()

    opt_factory = {"sgd": optim.SGD, "adam": optim.Adam, "rmsprop": optim.RMSprop}[opt_name.lower()]
    optimizer = opt_factory(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs)

    for _ in range(50):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    final_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            final_val_loss += loss.item()

    avg_val = final_val_loss / len(val_loader)
    if math.isnan(avg_val) or math.isinf(avg_val):
        avg_val = float('inf')
    return avg_val


def hyperparam_study(X, y):
    logging.info("Подбор гиперпараметров")
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y, test_size=0.2, val_size=0.2)
    in_features = X.shape[1]

    search_space = {
        "optimizer": ["SGD", "Adam", "RMSprop"],
        "lr": [1e-1, 1e-2, 1e-3],
        "bs": [16, 32, 64],
    }

    results = [
        {
            "optimizer": opt,
            "lr": lr,
            "batch_size": bs,
            "val_loss": _single_train_run(X_train, y_train, X_val, y_val, in_features, opt, lr, bs),
        }
        for opt in search_space["optimizer"]
        for lr in search_space["lr"]
        for bs in search_space["bs"]
    ]

    results_df = pd.DataFrame(results)
    # заменяем бесконечности на NaN
    results_df['val_loss'] = results_df['val_loss'].replace([np.inf, -np.inf], np.nan)
    logging.info("Итоговая таблица подбора:\n" + str(results_df))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_path = os.path.join(PLOTS_DIR, "hyperparam_results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"CSV с результатами сохранен: {csv_path}")

    plt.figure(figsize=(10, 6))
    for opt in results_df['optimizer'].unique():
        sub = results_df[results_df['optimizer'] == opt]
        sns.lineplot(data=sub, x='lr', y='val_loss', label=opt)
    plt.xscale('log')
    plt.title('Подбор гиперпараметров')
    plt.xlabel('Скорость обучения (log)')
    plt.ylabel('Ошибка (валидация)')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hyperparam_grid.png"))
    plt.close()

    # возвращаем лучший набор гиперпараметров
    best_cfg = results_df.loc[results_df['val_loss'].idxmin()].to_dict()
    logging.info(f"Лучшие гиперпараметры: {best_cfg}")
    return best_cfg


def feature_study(X, y, opt_name="adam", lr=1e-2, bs=32):
    logging.info("гененрация признаков")
    # Базовая модель
    logging.info("Учимся на исходных признаках")
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)
    baseline_loss = _single_train_run(X_train, y_train, X_val, y_val, n_in=X.shape[1], opt_name=opt_name, lr=lr, bs=bs)
    logging.info(f"Ошибка базовой модели (val): {baseline_loss:.4f}")

    # Инжиниринг признаков
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X.cpu().numpy())
    X_poly_t = torch.tensor(X_poly, dtype=torch.float32)

    logging.info(f"Было признаков: {X.shape[1]}, стало: {X_poly.shape[1]}")

    # Модель с новыми признаками
    logging.info("Учимся на расширенных признаках...")
    X_poly_train, X_poly_val, _, y_poly_train, y_poly_val, _ = split_data(X_poly_t, y)
    engineered_loss = _single_train_run(X_poly_train, y_poly_train, X_poly_val, y_poly_val, n_in=X_poly.shape[1],
                                        opt_name=opt_name, lr=lr, bs=bs)
    logging.info(f"Ошибка модели с новыми признаками на валидации: {engineered_loss:.4f}")

    if math.isinf(engineered_loss) or math.isnan(engineered_loss):
        logging.warning("poly-модель не сошлась; отображаем только baseline")
        engineered_loss = baseline_loss

    # Сравнение результатов
    plt.figure(figsize=(8, 6))
    plt.bar(['Базовая', 'Полиномиальные'], [baseline_loss, engineered_loss], color=['#ff7f0e', '#17becf'])
    plt.ylabel('Итоговая ошибка (валидация)')
    plt.title('Базовая vs. расширенная модель')
    plt.savefig(os.path.join(PLOTS_DIR, "feat_eng_compare.png"))
    plt.close()


def run_regression_experiment(csv_path: str, target: str = 'target', sample_frac: float = 1.0):
    os.makedirs(MODELS_DIR, exist_ok=True)
    dataset = CSVDataset(csv_path, target_col=target, task='regression', sample_frac=sample_frac)
    device = DEVICE
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LinearRegression(dataset.X.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    for epoch in range(50):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        log_epoch(epoch, 50, total_loss / len(loader))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{os.path.basename(csv_path)}_linreg.pt'))


def _single_train_run_cls(X_tr, y_tr, X_val, y_val, n_in, num_classes, opt_name="sgd", lr: float = 1e-2, bs: int = 32):
    """Один цикл обучения для SoftmaxRegression возвращает loss на валидации."""
    device = DEVICE
    model = SoftmaxRegression(n_in, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    opt_factory = {"sgd": optim.SGD, "adam": optim.Adam, "rmsprop": optim.RMSprop}[opt_name.lower()]
    optimizer = opt_factory(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs)
    for _ in range(30):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            total += criterion(model(xb), yb).item()
    avg_val = total / len(val_loader)
    if math.isnan(avg_val) or math.isinf(avg_val):
        avg_val = float('inf')
    return avg_val


def hyperparam_study_classification(X, y, num_classes):
    """Подбор гиперпараметров SoftmaxRegression, сохранение CSV, возврат лучшего набора."""
    logging.info("Подбор гиперпараметров (классификация)")
    X_tr, X_val, _, y_tr, y_val, _ = split_data(X, y, test_size=0.2, val_size=0.2)
    in_features = X.shape[1]
    search_space = {
        "optimizer": ["SGD", "Adam", "RMSprop"],
        "lr": [1e-1, 1e-2, 1e-3],
        "bs": [16, 32, 64],
    }
    results = [
        {"optimizer": opt, "lr": lr, "batch_size": bs,
         "val_loss": _single_train_run_cls(X_tr, y_tr, X_val, y_val, in_features, num_classes, opt, lr, bs)}
        for opt in search_space["optimizer"]
        for lr in search_space["lr"]
        for bs in search_space["bs"]
    ]
    df = pd.DataFrame(results)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_path = os.path.join(PLOTS_DIR, "hyperparam_results_cls.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"CSV классификационных гиперпараметров: {csv_path}")
    best_cfg = df.loc[df.val_loss.idxmin()].to_dict()
    logging.info(f"Лучшие (cls): {best_cfg}")
    return best_cfg


def run_classification_experiment(csv_path: str, target: str = 'label'):
    os.makedirs(MODELS_DIR, exist_ok=True)
    dataset = CSVDataset(csv_path, target_col=target, task='classification')
    device = DEVICE
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SoftmaxRegression(dataset.X.shape[1], dataset.num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(50):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        log_epoch(epoch, 50, total_loss / len(loader))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{os.path.basename(csv_path)}_logreg.pt'))


if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    heart_csv = os.path.join(BASE_DIR, 'data', 'heart.csv')
    house_csv = os.path.join(BASE_DIR, 'data', 'house_prices_train.csv')

    # Классификация: Heart Disease
    if os.path.exists(heart_csv):
        ds_heart = CSVDataset(heart_csv, target_col='target', task='classification')
        best_cls = hyperparam_study_classification(ds_heart.X, ds_heart.y, ds_heart.num_classes)
        run_classification_experiment(heart_csv, target='target')

    # Регрессия: House Prices
    if os.path.exists(house_csv):
        ds_house = CSVDataset(house_csv, target_col='SalePrice', task='regression', sample_frac=1.0)
        best_hp = hyperparam_study(ds_house.X, ds_house.y)
        feature_study(ds_house.X, ds_house.y,
                      opt_name=best_hp['optimizer'], lr=best_hp['lr'], bs=int(best_hp['batch_size']))
        run_regression_experiment(house_csv, target='SalePrice', sample_frac=1.0)