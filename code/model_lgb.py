from util import Util
import numpy as np
import lightgbm as lgb

class ModelLightGBM():
    def __init__(self):
        self.params = {
                        'learning_rate': 0.65,
                        'application': 'regression',
                        'max_depth': 3,
                        'num_leaves': 60,
                        'verbosity': -1,
                        'metric': 'RMSE',
                        'data_random_seed': 1,
                        'bagging_fraction': 0.5,
                        'nthread': 4,
                        'num_round': 10000
                      }
        self.model = None

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        
        # データのセット
        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y)
        if validation:
            lgb_valid = lgb.Dataset(va_x, va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')

        # 学習
        if validation:
            early_stopping_rounds = 20
            valid_sets = [lgb_train, lgb_valid]
            self.model = lgb.train(params, lgb_train, num_round,
                                   valid_names=["train", "valid"],
                                   valid_sets=valid_sets,
                                   early_stopping_rounds=early_stopping_rounds)
        else:
            self.model = lgb.train(params, dtrain, num_round,
                                   valid_names=["train"],
                                   valid_sets=[lgb_train])

    def predict(self, te_x):
        #lgb_test = lgb.Dataset(te_x)
        return self.model.predict(te_x)