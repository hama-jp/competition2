"""
Exp47: Leak-Free Pipeline + Exp07 Best Params
- Features: Same 31 features
- Target Encoding: Performed strictly INSIDE the CV loop (Leak-Free).
- Models: Cat, LGB, XGB using Exp07 parameters (which powered exp33).
- Goal: Recover LB performance while maintaining validation integrity.
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
from copy import deepcopy

warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

EXP_DIR = 'experiments/exp47_best_params_leak_free'
BASE_DIR = './'

os.makedirs(EXP_DIR, exist_ok=True)

print("=" * 60)
print("Exp47: Leak-Free Pipeline + Exp07 Params")
print("=" * 60)

# ==========================================
# Params (Restored from Exp07)
# ==========================================
cat_params = {
    "learning_rate": 0.07718772443488796,
    "depth": 3,
    "l2_leaf_reg": 0.0033458292447738312,
    "subsample": 0.8523245279212943,
    "min_data_in_leaf": 71,
    "random_strength": 1.2032200146196355,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "allow_writing_files": False,
    "iterations": 10000,
    "task_type": "CPU"
}

lgb_params = {
    "learning_rate": 0.09276457122109245,
    "num_leaves": 111,
    "max_depth": 3,
    "min_child_samples": 41,
    "subsample": 0.4126775090838999,
    "colsample_bytree": 0.4910286943468972,
    "reg_alpha": 0.5363517544276609,
    "reg_lambda": 9.847873834942789,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_estimators": 10000
}

xgb_params = {
    "learning_rate": 0.010557654198243605,
    "max_depth": 5,
    "min_child_weight": 2,
    "subsample": 0.42823789878654434,
    "colsample_bytree": 0.6400167112588607,
    "gamma": 0.7820326628699041,
    "alpha": 0.017218006622693675,
    "lambda": 0.0010527166617541805,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_estimators": 10000,
    "n_jobs": -1
}

# ==========================================
# Feature Loading & Base Prep (Same as exp46)
# ==========================================
# FEATURES WITHOUT _TE
base_features = [
    "Age_Year_Diff",
    "Broad_Jump_Type_Z",
    "School_Count",
    "Age_x_Momentum",
    "Momentum_Pos_Diff",
    "School",  # Need raw for TE
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
    "Position", # Need raw for TE
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
    "Position_Type", # Need raw for TE
    "Agility_3cone_Pos_Diff"
]

# Columns to TE
te_cols = ['School', 'Position', 'Position_Type']

# Final Feature List (after TE)
final_features = [f for f in base_features if f not in te_cols] + [f'{c}_TE' for c in te_cols]
print(f"Features: {len(final_features)}")

def perform_oof_te(train_df, test_df, col, target, n_folds=5, smoothing=10, seed=42):
    """
    Performs Target Encoding:
    - OOF for Train (to avoid self-target leak)
    - Mean for Test (using full Train)
    """
    global_mean = target.mean()
    train_encoded = np.zeros(len(train_df))
    
    # 1. OOF for Train
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(train_df, target):
        tr_data = train_df.iloc[tr_idx]
        tr_target = target.iloc[tr_idx]
        
        agg = tr_data.groupby(col).apply(lambda x: (
            (tr_target.loc[x.index].sum() + smoothing * global_mean) /
            (len(x) + smoothing)
        ))
        train_encoded[va_idx] = train_df.iloc[va_idx][col].map(agg).fillna(global_mean).values

    # 2. Mean for Test (using full train)
    agg_full = train_df.groupby(col).apply(lambda x: (
        (target.loc[x.index].sum() + smoothing * global_mean) /
        (len(x) + smoothing)
    ))
    test_encoded = test_df[col].map(agg_full).fillna(global_mean).values
    
    return train_encoded, test_encoded

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
    
    return train_df, test_df, target, cat_cols

print("\nLoading data...")
train_df, test_df, target, cat_cols = get_data()

cat_indices = [] # TE processed features are numerical

# ==========================================
# Prediction Holders
# ==========================================
oof_cat = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
pred_cat = np.zeros(len(test_df))
pred_lgb = np.zeros(len(test_df))
pred_xgb = np.zeros(len(test_df))

# ==========================================
# Main CV Loop (With TE Inside)
# ==========================================
print("\n--- Starting Leak-Free Training (Exp07 Params) ---")

for seed in SEEDS:
    print(f"\nSeed {seed}...")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, target)):
        # 1. Split Data
        X_tr_raw = train_df.iloc[tr_idx].copy()
        y_tr = target.iloc[tr_idx]
        X_va_raw = train_df.iloc[va_idx].copy()
        y_va = target.iloc[va_idx]
        X_te_raw = test_df.copy() # Use full test set each time
        
        # 2. Target Encoding INSIDE FOLD
        for col in te_cols:
            s_val = 20 if col == 'School' else (50 if col == 'Position' else 100)
            
            # Train (OOF)
            tr_enc, _ = perform_oof_te(X_tr_raw, X_tr_raw, col, y_tr, n_folds=5, smoothing=s_val, seed=42)
            X_tr_raw[f'{col}_TE'] = tr_enc
            
            # Valid & Test (Using Train Mean)
            global_mean = y_tr.mean()
            agg = X_tr_raw.groupby(col).apply(lambda x: (
                (y_tr.loc[x.index].sum() + s_val * global_mean) / (len(x) + s_val)
            ))
            X_va_raw[f'{col}_TE'] = X_va_raw[col].map(agg).fillna(global_mean)
            X_te_raw[f'{col}_TE'] = X_te_raw[col].map(agg).fillna(global_mean)

        # 3. Select Features
        X_tr = X_tr_raw[final_features]
        X_va = X_va_raw[final_features]
        X_te = X_te_raw[final_features]
        
        # 4. Train Models
        
        # --- CatBoost ---
        p = cat_params.copy()
        p['random_seed'] = seed
        model = CatBoostClassifier(**p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, verbose=False)
        oof_cat[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_cat += model.predict_proba(X_te)[:, 1] / (N_FOLDS * N_SEEDS)

        # --- LightGBM ---
        p = lgb_params.copy()
        p['seed'] = seed
        model = lgb.LGBMClassifier(**p)
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_names=['valid'], callbacks=callbacks)
        oof_lgb[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_lgb += model.predict_proba(X_te)[:, 1] / (N_FOLDS * N_SEEDS)

        # --- XGBoost ---
        p = xgb_params.copy()
        p['seed'] = seed
        p['early_stopping_rounds'] = 100
        model = xgb.XGBClassifier(**p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_xgb += model.predict_proba(X_te)[:, 1] / (N_FOLDS * N_SEEDS)

# ==========================================
# Optimization & Submission
# ==========================================
print("\n--- Calculating Ensemble ---")

# Calculate CVs
cv_cat = roc_auc_score(target, oof_cat)
cv_lgb = roc_auc_score(target, oof_lgb)
cv_xgb = roc_auc_score(target, oof_xgb)

print(f"CatBoost CV: {cv_cat:.5f}")
print(f"LightGBM CV: {cv_lgb:.5f}")
print(f"XGBoost CV:  {cv_xgb:.5f}")

def minimize_func(weights):
    final_oof = (weights[0] * oof_cat + weights[1] * oof_lgb + weights[2] * oof_xgb)
    return -roc_auc_score(target, final_oof)

init_weights = [0.4, 0.3, 0.3]
bounds = [(0, 1)] * 3
res = minimize(minimize_func, init_weights, bounds=bounds, method='SLSQP', constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
best_weights = res.x
final_cv = -res.fun

print(f"\nFinal Ensemble CV: {final_cv:.5f}")
print(f"Weights: Cat={best_weights[0]:.3f}, LGB={best_weights[1]:.3f}, XGB={best_weights[2]:.3f}")

pred_final = (best_weights[0] * pred_cat + best_weights[1] * pred_lgb + best_weights[2] * pred_xgb)

submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': pred_final})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

results = {
    'model': 'Exp47 Leak-Free + Exp07 Params',
    'cv_final': float(final_cv),
    'cv_individual': {'cat': float(cv_cat), 'lgb': float(cv_lgb), 'xgb': float(cv_xgb)},
    'weights': [float(w) for w in best_weights]
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved to {EXP_DIR}")
