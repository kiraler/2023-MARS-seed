import logging
import os
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LgbRegression():
    """回归问题的LGB模型，模型训练采用5折交叉验证，评估指标为MAE
    """
    def __init__(self, feats):
        self.k_fold = 5
        self.feats = feats
        self.abs_path = os.path.dirname(os.path.abspath(__file__))  # 程序运行路径
        self.models = {}

    def predict(self, df: pd.DataFrame, model_path=None):
        test_pred = np.zeros(len(df))
        logger.info(f'lgb输入数据大小: {df[self.feats].shape}')
        for fold_ in range(self.k_fold):
            if model_path is not None:
                lgb_model = lgb.Booster(model_file=os.path.join(model_path, f'lgb_model{fold_}.txt'))
            else:
                lgb_model = self.models[fold_]
            test_pred += lgb_model.predict(df[self.feats]) / self.k_fold
        logger.info('============模型预测完成============')

        # 模型输出
        df['pred'] = test_pred
        return df

    def train(self, df: pd.DataFrame, model_path=None, label='label', Stratifiedcol='') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """lgb模型训练

        Parameters
        ----------
        df : pd.DataFrame
            训练数据
        model_path : str, optional
            模型保存位置, by default ''
        label : str, optional
            标签列名, by default 'label'

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            包含预测结果的训练数据, 特征重要度
        """
        # folds = KFold(n_splits=self.k_fold, shuffle=True, random_state=2019)
        folds = StratifiedKFold(self.k_fold, shuffle=True, random_state=42)
        oof = np.zeros(df.shape[0])
        importance_lgb = 0
        # trn_df = df[df['ds'] < 20230408]
        # val_df = df[df['ds'] >= 20230408]
        # print(trn_df.shape, val_df.shape)
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(df, df[Stratifiedcol])):
        # for fold_ in range(1):
            print(f"========fold{fold_}========")
            X_trn, y_trn = df[self.feats].iloc[trn_idx], df[label].iloc[trn_idx]
            X_val, y_val = df[self.feats].iloc[val_idx], df[label].iloc[val_idx]
            # X_trn, y_trn = trn_df[self.feats], trn_df[label]
            # X_val, y_val = val_df[self.feats], val_df[label]
            trn_pred, val_pred, gbm = build_model_lgb_sklearn(X_trn, y_trn, X_val, y_val)

            oof[val_idx] = val_pred

            trn_score = np.sqrt(mean_squared_error(y_trn.values, trn_pred))
            val_score = np.sqrt(mean_squared_error(y_val.values, val_pred))
            print(f'trn rmse:  {trn_score:.3f}, val rmse:  {val_score:.3f}')

            importance_lgb += gbm.feature_importances_ / folds.n_splits
            if model_path is not None:
                gbm.booster_.save_model(os.path.join(model_path, f'lgb_model{fold_}.txt'))
            self.models[fold_] = gbm
# 
        oof_score = np.sqrt(mean_squared_error(df[label].values, oof))
        print(f'oof rmse:  {oof_score:.3f}')
        # 特征重要度
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = self.feats
        fold_importance_df["importance"] = importance_lgb
        fold_importance_df = fold_importance_df.sort_values(by='importance', ascending=False)

        # 模型输出
        df['pred'] = oof
        return df, fold_importance_df, np.round(oof_score, 1)

    def train_all(self, df, label, best_iter):
        model = LGBMRegressor(learning_rate=0.1,
                              boosting_type='gbdt',
                              n_estimators=best_iter,
                              objective='rmse',
                              subsample=0.6,
                              colsample_bytree=0.4,
                              num_leaves=100,
                              reg_lambda=3,
                              n_jobs=-1,
                              random_state=2020,
                              verbose=-1)
        model.fit(df[self.feats], df[label],
                  eval_metric='rmse',
                  callbacks=[lgb.log_evaluation(500)])
        return model

def build_model_lgb_sklearn(x_trn, y_trn, x_val, y_val):
    model = LGBMRegressor(learning_rate=0.1,
                          boosting_type='gbdt',
                          n_estimators=10000,
                          objective='rmse',
                          subsample=0.6,
                          colsample_bytree=0.4,
                          num_leaves=100,
                          reg_lambda=3,
                          n_jobs=-1,
                          random_state=2020,
                          verbose=-1)

    model.fit(x_trn,
              y_trn,
              eval_set=[(x_val, y_val)],
              eval_metric='rmse',
              callbacks=[
                  lgb.early_stopping(stopping_rounds=500),
                  lgb.log_evaluation(500)
              ])

    trn_pred = model.predict(x_trn)
    val_pred = model.predict(x_val)

    return trn_pred, val_pred, model
