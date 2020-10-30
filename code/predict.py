from model_nn import ModelMLP
from vectorizer import Vectorizer
from preprocessing import preprocess, title_torkenize, make_wakati
from load_data import load_data_from_gcs
import pandas as pd
import numpy as np
import re
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def main():
  with timer("model loading"):
    # モデルとパイプラインの読込
    model = ModelMLP()
    model.load_model()
    vectorizer = Vectorizer()
    vectorizer.load_vectorizer()

  with timer("data loading"):
    # 予測対象のデータをロード
    df = load_data_from_gcs()

  with timer("preprocess"):
    df = preprocess(df)


  with timer("predict"):
    X = df.drop(columns="price")
    X = vectorizer.transform(X)
    pred = model.predict(X)

    print(pred[:10])

if __name__ == "__main__":
  main()