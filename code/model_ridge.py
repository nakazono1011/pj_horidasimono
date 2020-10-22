from util import Util
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class ModelRidge():
    
    def __init__(self, run_name="Ridge"):
        self.run_name = run_name
        self.params = {
                      }
        self.model = None
        
    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        y_scaler   = StandardScaler() 
        tr_y_std = y_scaler.fit_transform(np.log1p(tr_y.values.reshape(-1, 1)))

        # モデルの構築
        model = Ridge()
        model.fit(tr_x, tr_y_std)

        # モデル・スケーラーの保持
        self.model = model
        self.scaler = y_scaler

    def predict(self, te_x):
        pred = self.model.predict(te_x).reshape(-1, 1)
        pred = self.scaler.inverse_transform(pred)[:, 0]
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
