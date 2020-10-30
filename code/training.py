from model_nn import ModelMLP
from vectorizer import Vectorizer
from preprocessing import preprocess, title_torkenize, make_wakati
from load_data import load_data_from_gcs
import pandas as pd
from contextlib import contextmanager
import time

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def main():
  # 学習データ読み込み
  with timer("train data load"):
    df = load_data_from_gcs()

  # 前処理
  with timer("preprocess"):
    df = preprocess(df)
    vectorizer = Vectorizer()

  X_train = df.drop(columns="price")
  y_train = df["price"]

  with timer("training"):
    X_train = vectorizer.fit_transform(X_train)

    # 学習
    base_params = {
                    'input_dropout': 0.2,
                    'hidden_layers': 3,
                    'hidden_units': 256,
                    'hidden_activation': 'relu',
                    'hidden_dropout': 0.2,
                    'batch_norm': 'before_act',
                    'optimizer': {'type': 'adam', 'lr': 5e-5},
                    'batch_size': 64,
                  }

    model = ModelMLP(base_params)
    model.fit(X_train, y_train)

  with timer("save model"):
    #モデルとパイプラインの保存
    vectorizer.save_vectorizer()
    model.save_model()

if __name__ == "__main__":
  main()