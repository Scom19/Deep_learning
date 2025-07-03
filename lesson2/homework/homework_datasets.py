from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import os
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CSVDataset(Dataset):
    def __init__(self, csv_file, target_col, task='regression', sample_frac: float = 1.0):
        self.task = task
        df = pd.read_csv(csv_file)
        # при необходимости уменьшаем число строк (Для увеличения скорости эксперемнтов для больших датасетов)
        if 0 < sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        numeric_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(exclude=['number']).columns

        # заполняем пропуски
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        if len(categorical_cols) > 0:
            X[categorical_cols] = X[categorical_cols].fillna('missing')
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.X = torch.nan_to_num(self.X)

        if self.task == 'regression':
            self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
            self.y = torch.nan_to_num(self.y)
        else:
            le = LabelEncoder()
            self.y = torch.tensor(le.fit_transform(y.values), dtype=torch.long)
            self.num_classes = len(le.classes_)
            self.class_names = le.classes_
        
        logging.info(
            f"Файл '{os.path.basename(csv_file)}' загружен: {len(self)} объектов, {self.X.shape[1]} атрибутов."
        )
        if self.task == 'classification':
            logging.info(f"Число классов: {self.num_classes}")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_linear_from_csv(csv_path: str, target_col: str, sample_frac: float = 1.0):
    from homework_model_modification import train_linear_regression_with_modifications
    from utils import split_data, plot_training_history

    ds = CSVDataset(csv_path, target_col, task='regression', sample_frac=sample_frac)
    X_tr, X_val, _, y_tr, y_val, _ = split_data(ds.X, ds.y)

    model, history = train_linear_regression_with_modifications(
        X_tr, y_tr, X_val, y_val, in_features=ds.X.shape[1]
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    weight_path = os.path.join(MODELS_DIR, f"{os.path.basename(csv_path)}_lin.pt")
    torch.save(model.state_dict(), weight_path)
    logging.info(f"Веса линейной модели сохранены в {weight_path}")

    plot_training_history(history, save_path=os.path.join(MODELS_DIR, f"{os.path.basename(csv_path)}_lin_plot.png"))


def train_logistic_from_csv(csv_path: str, target_col: str, sample_frac: float = 1.0):
    from homework_model_modification import train_softmax_regression
    from utils import split_data, plot_confusion_matrix

    ds = CSVDataset(csv_path, target_col, task='classification', sample_frac=sample_frac)
    X_tr, X_val, _, y_tr, y_val, _ = split_data(ds.X, ds.y)

    model, _, y_true, y_pred = train_softmax_regression(
        X_tr, y_tr, X_val, y_val,
        in_features=ds.X.shape[1], num_classes=ds.num_classes
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    weight_path = os.path.join(MODELS_DIR, f"{os.path.basename(csv_path)}_log.pt")
    torch.save(model.state_dict(), weight_path)
    logging.info(f"Веса логистической модели сохранены в {weight_path}")

    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(MODELS_DIR, f"{os.path.basename(csv_path)}_cm.png"))


if __name__ == "__main__":
    data_dir = os.path.join(BASE_DIR, 'data')
    titanic = os.path.join(data_dir, 'Titanic.csv')
    spotify = os.path.join(data_dir, 'spotify.csv')

    if os.path.exists(spotify):
        train_linear_from_csv(spotify, target_col='popularity', sample_frac=0.1)
    if os.path.exists(titanic):
        train_logistic_from_csv(titanic, target_col='Survived', sample_frac=1)