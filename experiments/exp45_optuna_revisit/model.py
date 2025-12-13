"""
Exp45: Re-optimize Hyperparameters for 31 features
- Base: exp44 features (31 features from exp33/exp13)
- Models: CatBoost, LightGBM, XGBoost
- Goal: Optimize hyperparameters specifically for this subset of features using Optuna.
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import optuna

# Reduce Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

N_TRIALS = 30  # Adjust as needed (e.g. 30-50 for quick check, 100+ for thorough)
N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

EXP_DIR = 'experiments/exp45_optuna_revisit'
BASE_DIR = './'

os.makedirs(EXP_DIR, exist_ok=True)

print("=" * 60)
print("Exp45: Re-optimization (Optuna) for 31 features")
print("=" * 60)

# ==========================================
# Feature Loading
# ==========================================
top30_features = [
    "School_TE",
    "Age_Year_Diff",
    "Broad_Jump_Type_Z",
    "School_Count",
    "Age_x_Momentum",
    "Momentum_Pos_Diff",
    "School",
    "Age_x_Speed",
    "Agility_3cone_Pos_Z",
    "Weight_Type_Z",
    "Bench_Press_Reps_Pos_Z",
    "Weight_Pos_Z",
    "Speed_Score_Year_Rank",
    "BMI_x_Speed",
    "Momentum_Type_Z",
    "BMI_Pos_Diff",
    "Age_div_Explosion",
    "Bench_Press_Reps_Pos_Diff",
    "Position_TE",
    "Work_Rate_Vertical_Type_Z",
    "Age",
    "Year",
    "Height_Pos_Diff",
    "Height_x_Weight",
    "BMI_Type_Z",
    "Agility_3cone_Type_Z",
    "Bench_per_Weight",
    "Sprint_Efficiency",
    "Broad_Jump_Year_Rank",
    "Position_Type_TE"
]
features = top30_features + ['Agility_3cone_Pos_Diff']
print(f"Features: {len(features)}")

# ==========================================
# Target Encoding
# ==========================================
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

# ==========================================
# Data Preparation
# ==========================================
def get_data():
    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

    train['is_train'] = 1
    test['is_train'] = 0
    test['Drafted'] = np.nan
    data = pd.concat([train, test], sort=False).reset_index(drop=True)

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

    data['Speed_x_Agility'] = data['Speed_Score'] * (1 / (data['Agility_Sum'] + 1))
    data['Power_x_Speed'] = data['Power_Sum'] * data['Speed_Score']
    data['BMI_x_Speed'] = data['BMI'] * data['Speed_Score']
    data['Weight_x_Vertical'] = data['Weight'] * data['Vertical_Jump']
    data['Height_x_Weight'] = data['Height'] * data['Weight']
    data['Age_Year_Diff'] = data['Age'] - data.groupby('Year')['Age'].transform('mean')
    data['Bench_per_Weight'] = data['Bench_Press_Reps'] * data['Weight'] / 100
    data['Jump_Efficiency'] = (data['Vertical_Jump'] + data['Broad_Jump']) / data['Weight']
    data['Sprint_Efficiency'] = data['Weight'] / (data['Sprint_40yd'] ** 2)

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

    rank_cols = ['Sprint_40yd', 'Vertical_Jump', 'Broad_Jump', 'Speed_Score', 'Explosion_Score']
    for col in rank_cols:
        ascending = col == 'Sprint_40yd'
        data[f'{col}_Pos_Rank'] = data.groupby('Position')[col].rank(ascending=ascending, pct=True)
        data[f'{col}_Year_Rank'] = data.groupby('Year')[col].rank(ascending=ascending, pct=True)

    data['School_Count'] = data['School'].map(data['School'].value_counts())
    data['School_Year_Count'] = data.groupby(['School', 'Year'])['Id'].transform('count')

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

    return train_df, test_df, target, cat_cols

# ==========================================
# Load data
# ==========================================
print("\nLoading data...")
train_df, test_df, target, cat_cols = get_data()
X_train = train_df[features]
y_train = target
X_test = test_df[features]

cat_indices = [features.index(c) for c in cat_cols if c in features]

# ==========================================
# Optimization Functions
# ==========================================

def optimize_cat(X, y, X_test, cat_indices):
    print("\n--- Optimizing CatBoost ---")
    
    def objective(trial):
        params = {
            'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': False, 'allow_writing_files': False,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'random_strength': trial.suggest_float('random_strength', 0, 5.0),
            'iterations': 1000  # Faster for optuna
        }
        
        # Use 5-fold CV for better accuracy
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=50, cat_features=cat_indices)
            scores.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"Best Cat CV: {study.best_value:.5f}")
    return study.best_params

def optimize_lgb(X, y, X_test):
    print("\n--- Optimizing LightGBM ---")
    
    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'verbosity': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'n_estimators': 1000
        }
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            model = lgb.LGBMClassifier(**params)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
            scores.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"Best LGB CV: {study.best_value:.5f}")
    return study.best_params

def optimize_xgb(X, y, X_test):
    print("\n--- Optimizing XGBoost ---")
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10.0),
            'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
            'n_estimators': 1000,
            'early_stopping_rounds': 50
        }
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            scores.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"Best XGB CV: {study.best_value:.5f}")
    return study.best_params

# ==========================================
# Run Optimization
# ==========================================
# ==========================================
# Run Optimization
# ==========================================
# cat_best = optimize_cat(X_train, y_train, X_test, cat_indices)
# lgb_best = optimize_lgb(X_train, y_train, X_test)
# xgb_best = optimize_xgb(X_train, y_train, X_test)

# Add fixed params back
# cat_best.update({'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': False, 'allow_writing_files': False, 'iterations': 10000, 'task_type': 'CPU'})
# lgb_best.update({'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'verbosity': -1, 'n_estimators': 10000})
# xgb_best.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'n_estimators': 10000, 'n_jobs': -1})

cat_best = {
    "learning_rate": 0.042416485432965945,
    "depth": 3,
    "l2_leaf_reg": 0.09319712200171228,
    "subsample": 0.7549829735745864,
    "min_data_in_leaf": 17,
    "random_strength": 1.8796757820025674,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "allow_writing_files": False,
    "iterations": 10000,
    "task_type": "CPU"
}

lgb_best = {
    "learning_rate": 0.08480189004395183,
    "num_leaves": 126,
    "max_depth": 3,
    "min_child_samples": 35,
    "subsample": 0.7809722220194688,
    "colsample_bytree": 0.7086994358713794,
    "reg_alpha": 0.059564041287579535,
    "reg_lambda": 0.011062670887356763,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_estimators": 10000
}

xgb_best = {
    "learning_rate": 0.04498719240007802,
    "max_depth": 5,  # Manually capped from 10 to 5 for safety
    "min_child_weight": 11,
    "subsample": 0.5096969812273043,
    "colsample_bytree": 0.7386532663298737,
    "gamma": 0.03962275519320668,
    "alpha": 0.0003260696374147742,
    "lambda": 0.00045918591375374273,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_estimators": 10000,
    "n_jobs": -1
}

print("\nBest Params (Manual Override):")
print("CAT:", cat_best)
print("LGB:", lgb_best)
print("XGB:", xgb_best)

# ==========================================
# Train Final Ensemble with Best Params
# ==========================================
print("\n--- Training Final Ensemble ---")
oof_cat = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
pred_cat = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))

for seed in SEEDS:
    print(f"Seed {seed}...")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
        
        # CAT
        p = cat_best.copy()
        p['random_seed'] = seed
        model = CatBoostClassifier(**p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, cat_features=cat_indices, verbose=False)
        oof_cat[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
        pred_cat += model.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

        # LGB
        p = lgb_best.copy()
        p['seed'] = seed
        model = lgb.LGBMClassifier(**p)
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_names=['valid'], callbacks=callbacks)
        oof_lgb[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
        pred_lgb += model.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

        # XGB
        p = xgb_best.copy()
        p['seed'] = seed
        p['early_stopping_rounds'] = 100
        model = xgb.XGBClassifier(**p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
        pred_xgb += model.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

# Ensemble Optimization
def minimize_func(weights):
    final_oof = (weights[0] * oof_cat + weights[1] * oof_lgb + weights[2] * oof_xgb)
    return -roc_auc_score(y_train, final_oof)

init_weights = [0.4, 0.3, 0.3]
bounds = [(0, 1)] * 3
res = minimize(minimize_func, init_weights, bounds=bounds, method='SLSQP', constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
best_weights = res.x

oof_final = (best_weights[0] * oof_cat + best_weights[1] * oof_lgb + best_weights[2] * oof_xgb)
pred_final = (best_weights[0] * pred_cat + best_weights[1] * pred_lgb + best_weights[2] * pred_xgb)
cv_final = roc_auc_score(y_train, oof_final)

print("\n" + "=" * 60)
print(f"Final Ensemble CV: {cv_final:.5f}")
print(f"Weights: Cat={best_weights[0]:.3f}, LGB={best_weights[1]:.3f}, XGB={best_weights[2]:.3f}")
print("=" * 60)

# Save
submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': pred_final})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

results = {
    'model': 'Re-optimized Ensemble',
    'cv_final': float(cv_final),
    'weights': [float(w) for w in best_weights],
    'params_cat': cat_best,
    'params_lgb': lgb_best,
    'params_xgb': xgb_best
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {EXP_DIR}")
