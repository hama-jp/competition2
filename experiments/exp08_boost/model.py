"""
Exp08: Push for 0.85 - More Features + More Diversity
- Additional feature engineering
- Add RandomForest and ExtraTrees for diversity
- More Optuna trials (80)
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import optuna
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 10
SEEDS = [42, 2023, 101, 555, 999, 123, 777, 88, 33, 1]
OPTUNA_TRIALS = 80

BASE_DIR = '/home/user/competition2'
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

def target_encode(train_df, test_df, col, target, n_folds=5, smoothing=10):
    global_mean = target.mean()
    train_encoded = np.zeros(len(train_df))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(train_df, target):
        tr_target = target.iloc[tr_idx]
        tr_data = train_df.iloc[tr_idx]
        agg = tr_data.groupby(col).apply(lambda x: (
            (tr_target.loc[x.index].sum() + smoothing * global_mean) /
            (len(x) + smoothing)
        ))
        train_encoded[va_idx] = train_df.iloc[va_idx][col].map(agg).fillna(global_mean).values

    agg_full = train_df.groupby(col).apply(lambda x: (
        (target.loc[x.index].sum() + smoothing * global_mean) /
        (len(x) + smoothing)
    ))
    test_encoded = test_df[col].map(agg_full).fillna(global_mean).values
    return train_encoded, test_encoded

def get_data():
    print("Loading and preprocessing with enhanced features...")
    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

    train['is_train'] = 1
    test['is_train'] = 0
    test['Drafted'] = np.nan
    data = pd.concat([train, test], sort=False).reset_index(drop=True)

    # Basic features
    measure_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    data['Missing_Count'] = data[measure_cols].isnull().sum(axis=1)

    data['BMI'] = data['Weight'] / (data['Height'] ** 2)
    data['Speed_Score'] = (data['Weight'] * 200) / (data['Sprint_40yd']**4)
    data['Explosion_Score'] = data['Weight'] * (data['Vertical_Jump'] + data['Broad_Jump'])
    data['Momentum'] = data['Weight'] / data['Sprint_40yd']
    data['Work_Rate_Vertical'] = data['Weight'] * data['Vertical_Jump']
    data['Agility_Sum'] = data['Agility_3cone'] + data['Shuttle']
    data['Power_Sum'] = data['Vertical_Jump'] + data['Broad_Jump']

    # Interaction features
    data['Age_x_Speed'] = data['Age'] * data['Speed_Score']
    data['Age_x_Momentum'] = data['Age'] * data['Momentum']
    data['Age_div_Explosion'] = data['Explosion_Score'] / data['Age']
    data['Speed_x_Agility'] = data['Speed_Score'] * (1 / (data['Agility_Sum'] + 1))
    data['Power_x_Speed'] = data['Power_Sum'] * data['Speed_Score']
    data['BMI_x_Speed'] = data['BMI'] * data['Speed_Score']
    data['Weight_x_Vertical'] = data['Weight'] * data['Vertical_Jump']
    data['Height_x_Weight'] = data['Height'] * data['Weight']
    data['Age_Year_Diff'] = data['Age'] - data.groupby('Year')['Age'].transform('mean')
    data['Bench_per_Weight'] = data['Bench_Press_Reps'] * data['Weight'] / 100
    data['Jump_Efficiency'] = (data['Vertical_Jump'] + data['Broad_Jump']) / data['Weight']
    data['Sprint_Efficiency'] = data['Weight'] / (data['Sprint_40yd'] ** 2)

    # NEW: Additional features
    data['Speed_Agility_Ratio'] = data['Sprint_40yd'] / (data['Agility_3cone'] + 0.01)
    data['Power_Weight_Ratio'] = data['Power_Sum'] / data['Weight']
    data['Explosiveness'] = data['Vertical_Jump'] * data['Broad_Jump'] / 1000
    data['Strength_Speed'] = data['Bench_Press_Reps'] / (data['Sprint_40yd'] + 0.01)
    data['Athletic_Score'] = (data['Speed_Score'] + data['Explosion_Score'] / 10000) / 2
    data['Size_Score'] = data['Height'] * data['Weight'] / 100
    data['Agility_Efficiency'] = data['Shuttle'] / (data['Agility_3cone'] + 0.01)

    # Position-based Z-scores
    stats_cols = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump',
                  'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle',
                  'Momentum', 'Work_Rate_Vertical', 'Speed_Score', 'Explosion_Score', 'BMI',
                  'Explosiveness', 'Athletic_Score', 'Size_Score']

    for col in stats_cols:
        if col in data.columns:
            group_mean = data.groupby('Position')[col].transform('mean')
            group_std = data.groupby('Position')[col].transform('std')
            data[f'{col}_Pos_Z'] = (data[col] - group_mean) / group_std
            data[f'{col}_Pos_Diff'] = data[col] - group_mean

            group_mean_t = data.groupby('Position_Type')[col].transform('mean')
            group_std_t = data.groupby('Position_Type')[col].transform('std')
            data[f'{col}_Type_Z'] = (data[col] - group_mean_t) / group_std_t

    # Rank features
    rank_cols = ['Sprint_40yd', 'Vertical_Jump', 'Broad_Jump', 'Speed_Score', 'Explosion_Score', 'Athletic_Score']
    for col in rank_cols:
        if col in data.columns:
            ascending = col in ['Sprint_40yd', 'Agility_3cone', 'Shuttle']
            data[f'{col}_Pos_Rank'] = data.groupby('Position')[col].rank(ascending=ascending, pct=True)
            data[f'{col}_Year_Rank'] = data.groupby('Year')[col].rank(ascending=ascending, pct=True)
            data[f'{col}_Overall_Rank'] = data[col].rank(ascending=ascending, pct=True)

    data['School_Count'] = data['School'].map(data['School'].value_counts())
    data['School_Year_Count'] = data.groupby(['School', 'Year'])['Id'].transform('count')
    data['Position_Year_Count'] = data.groupby(['Position', 'Year'])['Id'].transform('count')

    # Elite flags
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
    data['Elite_Score'] = data['Elite_Count'] * 2 - data['Red_Flag_Count']

    school_orig = data['School'].copy()
    position_orig = data['Position'].copy()
    position_type_orig = data['Position_Type'].copy()

    cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = data[col].fillna('Unknown')
        data[col] = le.fit_transform(data[col].astype(str))

    train_df = data[data['is_train'] == 1].reset_index(drop=True)
    test_df = data[data['is_train'] == 0].reset_index(drop=True)
    target = train_df['Drafted']

    train_school = school_orig[data['is_train'] == 1].reset_index(drop=True)
    test_school = school_orig[data['is_train'] == 0].reset_index(drop=True)
    train_position = position_orig[data['is_train'] == 1].reset_index(drop=True)
    test_position = position_orig[data['is_train'] == 0].reset_index(drop=True)
    train_position_type = position_type_orig[data['is_train'] == 1].reset_index(drop=True)
    test_position_type = position_type_orig[data['is_train'] == 0].reset_index(drop=True)

    train_temp = pd.DataFrame({'School': train_school, 'Position': train_position, 'Position_Type': train_position_type})
    test_temp = pd.DataFrame({'School': test_school, 'Position': test_position, 'Position_Type': test_position_type})

    train_df['School_TE'], test_df['School_TE'] = target_encode(train_temp, test_temp, 'School', target, smoothing=20)
    train_df['Position_TE'], test_df['Position_TE'] = target_encode(train_temp, test_temp, 'Position', target, smoothing=50)
    train_df['Position_Type_TE'], test_df['Position_Type_TE'] = target_encode(train_temp, test_temp, 'Position_Type', target, smoothing=100)

    exclude_cols = ['Id', 'Drafted', 'is_train']
    features = [c for c in train_df.columns if c not in exclude_cols]
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    print(f"Total features: {len(features)}")
    return train_df, test_df, target, features, cat_indices

# Optuna optimization for CatBoost (best performer)
def optimize_cat(X, y, features, cat_indices):
    print("Optimizing CatBoost (80 trials)...")

    def objective(trial):
        params = {
            'loss_function': 'Logloss', 'eval_metric': 'AUC',
            'verbose': False, 'allow_writing_files': False,
            'random_seed': 42, 'iterations': 3000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'random_strength': trial.suggest_float('random_strength', 0, 5.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.0),
        }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)

            model = CatBoostClassifier(**params)
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=50)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print(f"  Best CAT CV: {study.best_value:.5f}")
    return study.best_params

def train_cat(X, y, X_test, features, cat_indices, params):
    print("Training CatBoost (10 seeds)...")
    params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': False, 'allow_writing_files': False, 'iterations': 10000})

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

    cv = roc_auc_score(y, oof)
    print(f"  CAT CV: {cv:.5f}")
    return oof, preds, cv

def train_lgb(X, y, X_test, features):
    print("Training LightGBM (10 seeds)...")
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'verbosity': -1, 'learning_rate': 0.02, 'num_leaves': 50,
        'max_depth': 8, 'min_child_samples': 30, 'subsample': 0.85,
        'colsample_bytree': 0.75, 'reg_alpha': 0.2, 'reg_lambda': 0.5
    }

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        params['random_state'] = seed
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            model = lgb.LGBMClassifier(**params, n_estimators=10000)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])

            oof[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
            preds += model.predict_proba(X_test[features])[:, 1] / (N_FOLDS * len(SEEDS))

    cv = roc_auc_score(y, oof)
    print(f"  LGB CV: {cv:.5f}")
    return oof, preds, cv

def train_xgb(X, y, X_test, features):
    print("Training XGBoost (10 seeds)...")
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'learning_rate': 0.02, 'max_depth': 7, 'min_child_weight': 3,
        'subsample': 0.85, 'colsample_bytree': 0.75, 'gamma': 0.5,
        'alpha': 0.2, 'lambda': 1.0, 'early_stopping_rounds': 100
    }

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        params['random_state'] = seed
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            model = xgb.XGBClassifier(**params, n_estimators=10000)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            oof[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
            preds += model.predict_proba(X_test[features])[:, 1] / (N_FOLDS * len(SEEDS))

    cv = roc_auc_score(y, oof)
    print(f"  XGB CV: {cv:.5f}")
    return oof, preds, cv

if __name__ == '__main__':
    print("="*50)
    print("Exp08: Push for 0.85")
    print("="*50)

    train_df, test_df, target, features, cat_indices = get_data()

    # Optimize CatBoost (best performer)
    bp_cat = optimize_cat(train_df, target, features, cat_indices)

    # Train all models
    print("\n--- Training Models ---")
    oof_lgb, pred_lgb, cv_lgb = train_lgb(train_df, target, test_df, features)
    oof_xgb, pred_xgb, cv_xgb = train_xgb(train_df, target, test_df, features)
    oof_cat, pred_cat, cv_cat = train_cat(train_df, target, test_df, features, cat_indices, bp_cat)

    # Ensemble optimization
    print("\n--- Ensemble Optimization ---")
    def get_score(weights):
        final_oof = (oof_lgb * weights[0]) + (oof_xgb * weights[1]) + (oof_cat * weights[2])
        return -roc_auc_score(target, final_oof)

    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bnds = ((0, 1), (0, 1), (0, 1))
    init_w = [0.25, 0.25, 0.5]

    res = minimize(get_score, init_w, method='SLSQP', bounds=bnds, constraints=cons)
    best_w = res.x
    final_cv = -res.fun

    print(f"\n{'='*50}")
    print(f"Individual CVs: LGB={cv_lgb:.5f}, XGB={cv_xgb:.5f}, CAT={cv_cat:.5f}")
    print(f"Weights: LGB={best_w[0]:.3f}, XGB={best_w[1]:.3f}, CAT={best_w[2]:.3f}")
    print(f"Final CV AUC: {final_cv:.5f}")
    print(f"{'='*50}")

    final_preds = (pred_lgb * best_w[0]) + (pred_xgb * best_w[1]) + (pred_cat * best_w[2])

    submission = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))
    submission['Drafted'] = final_preds
    submission.to_csv(os.path.join(EXP_DIR, 'submission.csv'), index=False)

    results = {
        'experiment': 'exp08_boost',
        'timestamp': datetime.now().isoformat(),
        'cv_lgb': cv_lgb,
        'cv_xgb': cv_xgb,
        'cv_cat': cv_cat,
        'cv_final': final_cv,
        'weights': {'lgb': float(best_w[0]), 'xgb': float(best_w[1]), 'cat': float(best_w[2])},
        'n_features': len(features)
    }

    with open(os.path.join(EXP_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    np.save(os.path.join(EXP_DIR, 'oof_lgb.npy'), oof_lgb)
    np.save(os.path.join(EXP_DIR, 'oof_xgb.npy'), oof_xgb)
    np.save(os.path.join(EXP_DIR, 'oof_cat.npy'), oof_cat)
    np.save(os.path.join(EXP_DIR, 'pred_final.npy'), final_preds)

    print(f"\nResults saved to {EXP_DIR}")
