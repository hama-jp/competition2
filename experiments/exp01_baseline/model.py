"""
Exp01: Baseline Model
現在のmodel_v01.pyをベースに、シード数を減らして迅速にCVを確認
"""
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings('ignore')

# 定数
N_FOLDS = 5
N_SEEDS = 3  # 時間短縮のため3に削減
SEEDS = [42, 2023, 101]

# パス設定
BASE_DIR = '/home/user/competition2'
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. データ準備
# ==========================================
SELECTED_FEATURES = [
    'Age', 'Speed_Score_Pos_Z', 'Age_div_Explosion', 'Speed_Score_Type_Z',
    'Speed_Score_Pos_Diff', 'Sprint_40yd_Pos_Z', 'Speed_Score', 'Age_x_Momentum',
    'Age_x_Speed', 'Momentum_Pos_Z', 'Player_Type', 'Explosion_Score',
    'Agility_3cone_Pos_Diff', 'Work_Rate_Vertical', 'Sprint_40yd_Type_Z', 'Shuttle',
    'Broad_Jump_Pos_Z', 'Weight', 'Sprint_40yd', 'Sprint_40yd_Pos_Diff',
    'Momentum', 'Position', 'Weight_Pos_Z', 'Work_Rate_Vertical_Pos_Diff',
    'School_Count', 'Height', 'BMI_Pos_Z', 'Momentum_Pos_Diff',
    'Agility_3cone_Pos_Z', 'Agility_3cone_Type_Z', 'Vertical_Jump', 'Year',
    'Bench_Press_Reps_Pos_Diff', 'Explosion_Score_Pos_Diff', 'Explosion_Score_Type_Z',
    'Height_Pos_Diff', 'Shuttle_Pos_Z', 'Vertical_Jump_Type_Z', 'Weight_Type_Z',
    'Agility_3cone', 'Bench_Press_Reps', 'BMI_Pos_Diff', 'Work_Rate_Vertical_Pos_Z',
    'Broad_Jump', 'Shuttle_Type_Z', 'Power_Sum', 'Position_Type', 'Missing_Count',
    'Broad_Jump_Type_Z', 'Weight_Pos_Diff', 'Explosion_Score_Pos_Z', 'Shuttle_Pos_Diff', 'School',
    'Elite_Count', 'Red_Flag_Count', 'Talent_Diff'
]

def get_data():
    print("Loading and preprocessing...")
    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

    train['is_train'] = 1
    test['is_train'] = 0
    test['Drafted'] = np.nan
    data = pd.concat([train, test], sort=False).reset_index(drop=True)

    # --- Feature Engineering ---
    measure_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    data['Missing_Count'] = data[measure_cols].isnull().sum(axis=1)

    data['BMI'] = data['Weight'] / (data['Height'] ** 2)
    data['Speed_Score'] = (data['Weight'] * 200) / (data['Sprint_40yd']**4)
    data['Explosion_Score'] = data['Weight'] * (data['Vertical_Jump'] + data['Broad_Jump'])
    data['Momentum'] = data['Weight'] / data['Sprint_40yd']
    data['Work_Rate_Vertical'] = data['Weight'] * data['Vertical_Jump']
    data['Agility_Sum'] = data['Agility_3cone'] + data['Shuttle']
    data['Power_Sum'] = data['Vertical_Jump'] + data['Broad_Jump']

    data['Age_x_Speed'] = data['Age'] * data['Speed_Score']
    data['Age_x_Momentum'] = data['Age'] * data['Momentum']
    data['Age_div_Explosion'] = data['Explosion_Score'] / data['Age']

    stats_cols = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump',
                  'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle',
                  'Momentum', 'Work_Rate_Vertical', 'Speed_Score', 'Explosion_Score', 'BMI']

    for col in stats_cols:
        group_mean = data.groupby('Position')[col].transform('mean')
        group_std = data.groupby('Position')[col].transform('std')
        data[f'{col}_Pos_Z'] = (data[col] - group_mean) / group_std
        data[f'{col}_Pos_Diff'] = data[col] - group_mean

        group_mean_t = data.groupby('Position_Type')[col].transform('mean')
        group_std_t = data.groupby('Position_Type')[col].transform('std')
        data[f'{col}_Type_Z'] = (data[col] - group_mean_t) / group_std_t

    data['School_Count'] = data['School'].map(data['School'].value_counts())

    cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = data[col].fillna('Unknown')
        data[col] = le.fit_transform(data[col].astype(str))

    # --- Elite & Red Flags ---
    phys_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    lower_is_better = ['Sprint_40yd', 'Agility_3cone', 'Shuttle']

    elite_flags = pd.DataFrame(index=data.index)
    red_flags = pd.DataFrame(index=data.index)

    for col in phys_cols:
        if col in lower_is_better:
            q10 = data.groupby('Position')[col].transform(lambda x: x.quantile(0.1))
            q90 = data.groupby('Position')[col].transform(lambda x: x.quantile(0.9))
            elite_flags[f'{col}_Elite'] = (data[col] <= q10).astype(int)
            red_flags[f'{col}_Bad'] = (data[col] >= q90).astype(int)
        else:
            q90 = data.groupby('Position')[col].transform(lambda x: x.quantile(0.9))
            q10 = data.groupby('Position')[col].transform(lambda x: x.quantile(0.1))
            elite_flags[f'{col}_Elite'] = (data[col] >= q90).astype(int)
            red_flags[f'{col}_Bad'] = (data[col] <= q10).astype(int)

    data['Elite_Count'] = elite_flags.sum(axis=1)
    data['Red_Flag_Count'] = red_flags.sum(axis=1)
    data['Talent_Diff'] = data['Elite_Count'] - data['Red_Flag_Count']

    # Features Filtering
    cols = ['Id', 'Drafted', 'is_train'] + SELECTED_FEATURES
    existing_cols = [c for c in cols if c in data.columns]
    data = data[existing_cols]

    features = [c for c in data.columns if c not in ['Id', 'Drafted', 'is_train']]
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    train_df = data[data['is_train'] == 1].reset_index(drop=True)
    test_df = data[data['is_train'] == 0].reset_index(drop=True)
    target = train_df['Drafted']

    return train_df, test_df, target, features, cat_indices

# ==========================================
# 2. Training Functions
# ==========================================
def train_lgb(X, y, X_test, features, params=None):
    if params is None:
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'verbosity': -1, 'learning_rate': 0.03, 'num_leaves': 50,
            'max_depth': 6, 'min_child_samples': 30, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1
        }

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        params['random_state'] = seed
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            model = lgb.LGBMClassifier(**params, n_estimators=5000)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])

            oof[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
            preds += model.predict_proba(X_test[features])[:, 1] / (N_FOLDS * len(SEEDS))

    print(f"LGB CV: {roc_auc_score(y, oof):.5f}")
    return oof, preds

def train_xgb(X, y, X_test, features, params=None):
    if params is None:
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
            'learning_rate': 0.03, 'max_depth': 6, 'min_child_weight': 5,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.5,
            'alpha': 0.1, 'lambda': 0.1, 'early_stopping_rounds': 100
        }

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        params['random_state'] = seed
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            model = xgb.XGBClassifier(**params, n_estimators=5000)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            oof[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
            preds += model.predict_proba(X_test[features])[:, 1] / (N_FOLDS * len(SEEDS))

    print(f"XGB CV: {roc_auc_score(y, oof):.5f}")
    return oof, preds

def train_cat(X, y, X_test, features, cat_indices, params=None):
    if params is None:
        params = {
            'loss_function': 'Logloss', 'eval_metric': 'AUC',
            'verbose': False, 'allow_writing_files': False,
            'iterations': 5000, 'learning_rate': 0.03, 'depth': 6,
            'l2_leaf_reg': 3.0, 'subsample': 0.8, 'min_data_in_leaf': 30
        }

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        params['random_seed'] = seed
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)

            model = CatBoostClassifier(**params)
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

            oof[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
            preds += model.predict_proba(X_test[features])[:, 1] / (N_FOLDS * len(SEEDS))

    print(f"CAT CV: {roc_auc_score(y, oof):.5f}")
    return oof, preds

# ==========================================
# Main Pipeline
# ==========================================
if __name__ == '__main__':
    print("="*50)
    print("Exp01: Baseline Model")
    print("="*50)

    train_df, test_df, target, features, cat_indices = get_data()
    print(f"Features: {len(features)}")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Train models
    oof_lgb, pred_lgb = train_lgb(train_df, target, test_df, features)
    oof_xgb, pred_xgb = train_xgb(train_df, target, test_df, features)
    oof_cat, pred_cat = train_cat(train_df, target, test_df, features, cat_indices)

    # Ensemble optimization
    def get_score(weights):
        final_oof = (oof_lgb * weights[0]) + (oof_xgb * weights[1]) + (oof_cat * weights[2])
        return -roc_auc_score(target, final_oof)

    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bnds = ((0, 1), (0, 1), (0, 1))
    init_w = [0.3, 0.3, 0.4]

    res = minimize(get_score, init_w, method='SLSQP', bounds=bnds, constraints=cons)
    best_w = res.x

    final_cv = -res.fun
    print(f"\n{'='*50}")
    print(f"Weights: LGB={best_w[0]:.3f}, XGB={best_w[1]:.3f}, CAT={best_w[2]:.3f}")
    print(f"Final CV AUC: {final_cv:.5f}")
    print(f"{'='*50}")

    # Save predictions
    final_preds = (pred_lgb * best_w[0]) + (pred_xgb * best_w[1]) + (pred_cat * best_w[2])

    submission = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))
    submission['Drafted'] = final_preds
    submission.to_csv(os.path.join(EXP_DIR, 'submission.csv'), index=False)

    # Save results
    results = {
        'experiment': 'exp01_baseline',
        'timestamp': datetime.now().isoformat(),
        'cv_lgb': roc_auc_score(target, oof_lgb),
        'cv_xgb': roc_auc_score(target, oof_xgb),
        'cv_cat': roc_auc_score(target, oof_cat),
        'cv_final': final_cv,
        'weights': {'lgb': best_w[0], 'xgb': best_w[1], 'cat': best_w[2]},
        'n_features': len(features),
        'n_seeds': len(SEEDS)
    }

    with open(os.path.join(EXP_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save OOF predictions for stacking
    np.save(os.path.join(EXP_DIR, 'oof_lgb.npy'), oof_lgb)
    np.save(os.path.join(EXP_DIR, 'oof_xgb.npy'), oof_xgb)
    np.save(os.path.join(EXP_DIR, 'oof_cat.npy'), oof_cat)
    np.save(os.path.join(EXP_DIR, 'pred_lgb.npy'), pred_lgb)
    np.save(os.path.join(EXP_DIR, 'pred_xgb.npy'), pred_xgb)
    np.save(os.path.join(EXP_DIR, 'pred_cat.npy'), pred_cat)

    print(f"\nResults saved to {EXP_DIR}")
