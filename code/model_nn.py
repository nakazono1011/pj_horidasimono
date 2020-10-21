from util import Util
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.simplefilter("ignore")

from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils

import numpy as np

class ModelNN():
    
    def __init__(self, vectorizer, run_name="NN"):
        self.run_name = run_name
        self.params = {
                      }
        self.model = None
        self.vectorizer = vectorizer

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        tr_x = self.vectorizer.fit_transform(tr_x).astype(np.float32)
        
        y_scaler   = StandardScaler() 
        tr_y_std = y_scaler.fit_transform(np.log1p(tr_y.values.reshape(-1, 1)))

        # データのセット・スケーリング
        validation = va_x is not None
        if validation:
            va_x = self.vectorizer.transform(va_x).astype(np.float32)
            va_y_std = y_scaler.transform(np.log1p(va_y.values.reshape(-1, 1)))

        # モデルの構築
        model = Sequential()
        model.add(Dense(512, activation='relu' ,input_shape=(tr_x.shape[1],)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="linear"))
        model.compile(loss='mean_squared_error', optimizer='adam')

        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                                           verbose=1, restore_best_weights=True)
            model.fit(tr_x, tr_y_std, epochs=200, batch_size=512, verbose=1,
                      validation_data=(va_x, va_y_std), callbacks=[early_stopping])
        else:
            model.fit(tr_x, tr_y_std, epochs=200, batch_size=512, verbose=1)

        # モデル・スケーラーの保持
        self.model = model
        self.scaler = y_scaler

    def predict(self, te_x):
        te_x = self.vectorizer.transform(te_x)
        pred = self.model.predict(te_x).reshape(-1,1)
        pred = self.scaler.inverse_transform(pred)[:,0]
        return pred
    
    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_name}.h5')
        scaler_path = os.path.join('../model/model', f'{self.run_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_name}.h5')
        scaler_path = os.path.join('../model/model', f'{self.run_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)
