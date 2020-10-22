from util import Util
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter("ignore")

from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU, ReLU
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import backend as K

import numpy as np

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


class ModelNN():
    
    def __init__(self, params, vectorizer, run_name="NN"):
        self.run_name = run_name
        self.params = params
        self.model = None

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):

        # パラメータ
        input_dropout = self.params['input_dropout']
        hidden_layers = int(self.params['hidden_layers'])
        hidden_units = int(self.params['hidden_units'])
        hidden_activation = self.params['hidden_activation']
        hidden_dropout = self.params['hidden_dropout']
        batch_norm = self.params['batch_norm']
        optimizer_type = self.params['optimizer']['type']
        optimizer_lr = self.params['optimizer']['lr']
        batch_size = int(self.params['batch_size'])

        #ベクトライズ・標準化
        y_scaler   = StandardScaler() 
        tr_y_std = y_scaler.fit_transform(np.log1p(tr_y.values.reshape(-1, 1)))

        # データのセット・スケーリング
        validation = va_x is not None
        if validation:
            va_x = self.vectorizer.transform(va_x).astype(np.float32)
            va_y_std = y_scaler.transform(np.log1p(va_y.values.reshape(-1, 1)))

        # モデルの構築
        # 入力層
        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(tr_x.shape[1],)))
        self.model.add(PReLU())
        self.model.add(Dropout(input_dropout))

        # 中間層
        for i in range(hidden_layers):
            self.model.add(Dense(hidden_units))
            if batch_norm == 'before_act':
                self.model.add(BatchNormalization())
            if hidden_activation == 'prelu':
                self.model.add(PReLU())
            elif hidden_activation == 'relu':
                self.model.add(ReLU())
            else:
                raise NotImplementedError
            self.model.add(Dropout(hidden_dropout))
        
        self.model.add(Dense(1, activation="linear"))

        # オプティマイザ
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=optimizer_lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimizer_type == 'adam':
            optimizer = Adam(lr=optimizer_lr, beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError

        # 目的関数、評価指標などの設定    
        self.model.compile(loss=root_mean_squared_error, optimizer=optimizer)

        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                                           verbose=1, restore_best_weights=True)
            self.model.fit(tr_x, tr_y_std, epochs=200, batch_size=64, verbose=2,
                      validation_data=(va_x, va_y_std), callbacks=[early_stopping])
        else:
            self.model.fit(tr_x, tr_y_std, epochs=200, batch_size=64, verbose=2)

        # モデル・スケーラーの保持
        self.scaler = y_scaler

    def predict(self, te_x):
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
