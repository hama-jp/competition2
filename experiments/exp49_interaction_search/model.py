"""
Exp49: Interaction Search
- Base: Exp48 (Leak-Free + Unsupervised)
- New Features: Systematic pairwise interactions (Product/Ratio) of 8 physical stats.
- Method:
  - Row-wise interactions (A*B, A/B) are generated globally (Safe).
  - Unsupervised & TE are still fold-based.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from copy import deepcopy
from itertools import permutations

warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

EXP_DIR = 'experiments/exp49_interaction_search'
BASE_DIR = './'

os.makedirs(EXP_DIR, exist_ok=True)

print("=" * 60)
print("Exp49: Interaction Search")
print("=" * 60)

# ==========================================
# Params (Same as Exp48)
# ==========================================
cat_params = {
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

lgb_params = {
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

xgb_params = {
    "learning_rate": 0.04498719240007802,
    "max_depth": 5,
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

# ==========================================
# Feature Loading & Base Prep
# ==========================================
base_features = [
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
    "Position",
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
    "Position_Type",
    "Agility_3cone_Pos_Diff"
]

te_cols = ['School', 'Position', 'Position_Type']
phys_cols = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']

def perform_oof_te(train_df, test_df, col, target, n_folds=5, smoothing=10, seed=42):
    global_mean = target.mean()
    train_encoded = np.zeros(len(train_df))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(train_df, target):
        tr_data = train_df.iloc[tr_idx]
        tr_target = target.iloc[tr_idx]
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
    
    phys_cols_calc = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    lower_is_better = ['Sprint_40yd', 'Agility_3cone', 'Shuttle']
    elite_flags = pd.DataFrame(index=data.index)
    red_flags = pd.DataFrame(index=data.index)
    for col in phys_cols_calc:
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
    
    cat_cols_enc = ['School', 'Player_Type', 'Position_Type', 'Position']
    for col in cat_cols_enc:
        le = LabelEncoder()
        data[col] = data[col].fillna('Unknown')
        data[col] = le.fit_transform(data[col].astype(str))
    
    # --- Generate Interactions (Safe to do globally as it is row-wise) ---
    print("Generating Interactions...")
    interaction_feats = []
    
    # Generate Product and Ratio for all pairs
    # Note: Using itertools.permutations for div (A/B and B/A), and combinations for mult (A*B same as B*A)
    # Actually, simpler to just double loop.
    for i, col1 in enumerate(phys_cols):
        for j, col2 in enumerate(phys_cols):
            if i < j:
                # Product
                f_name = f'{col1}_x_{col2}'
                data[f_name] = data[col1] * data[col2]
                if f_name not in base_features:
                    interaction_feats.append(f_name)
            
            if i != j:
                # Ratio (col1 / col2)
                # Handle div by zero safely (add epsilon or np.inf -> nan -> impute later)
                f_name = f'{col1}_div_{col2}'
                # If col2 is 0 or nan, result is nan/inf. Tree models handle this, but let's be cleaner.
                # Replace 0 with small epsilon? Or just let it be inf.
                # Just dividing.
                data[f_name] = data[col1] / (data[col2] + 1e-6)
                if f_name not in base_features:
                    interaction_feats.append(f_name)
    
    print(f"Generated {len(interaction_feats)} interaction features.")
        
    train_df = data[data['is_train'] == 1].reset_index(drop=True)
    test_df = data[data['is_train'] == 0].reset_index(drop=True)
    target = train_df['Drafted']
    return train_df, test_df, target, interaction_feats

print("\nLoading data...")
train_df, test_df, target, interaction_feats = get_data()

# ==========================================
# Main CV Loop
# ==========================================
print("\n--- Starting Leak-Free Training (Interactions included) ---")

oof_cat = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
pred_cat = np.zeros(len(test_df))
pred_lgb = np.zeros(len(test_df))
pred_xgb = np.zeros(len(test_df))

for seed in SEEDS:
    print(f"\nSeed {seed}...")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, target)):
        # 1. Split Data
        X_tr_raw = train_df.iloc[tr_idx].copy()
        y_tr = target.iloc[tr_idx]
        X_va_raw = train_df.iloc[va_idx].copy()
        y_va = target.iloc[va_idx]
        X_te_raw = test_df.copy()
        
        # 2. Target Encoding
        for col in te_cols:
            s_val = 20 if col == 'School' else (50 if col == 'Position' else 100)
            tr_enc, _ = perform_oof_te(X_tr_raw, X_tr_raw, col, y_tr, n_folds=5, smoothing=s_val, seed=42)
            X_tr_raw[f'{col}_TE'] = tr_enc
            global_mean = y_tr.mean()
            agg = X_tr_raw.groupby(col).apply(lambda x: ((y_tr.loc[x.index].sum() + s_val * global_mean) / (len(x) + s_val)))
            X_va_raw[f'{col}_TE'] = X_va_raw[col].map(agg).fillna(global_mean)
            X_te_raw[f'{col}_TE'] = X_te_raw[col].map(agg).fillna(global_mean)
            
        # 3. Unsupervised Features (Impute -> Scale -> Transform)
        X_tr_phys = X_tr_raw[phys_cols].copy()
        X_va_phys = X_va_raw[phys_cols].copy()
        X_te_phys = X_te_raw[phys_cols].copy()
        
        imputer = SimpleImputer(strategy='median')
        X_tr_phys_imp = imputer.fit_transform(X_tr_phys)
        X_va_phys_imp = imputer.transform(X_va_phys)
        X_te_phys_imp = imputer.transform(X_te_phys)
        
        scaler = StandardScaler()
        X_tr_phys_sc = scaler.fit_transform(X_tr_phys_imp)
        X_va_phys_sc = scaler.transform(X_va_phys_imp)
        X_te_phys_sc = scaler.transform(X_te_phys_imp)
        
        for k in [3, 6, 12, 50]:
            kmeans = KMeans(n_clusters=k, random_state=seed)
            kmeans.fit(X_tr_phys_sc)
            X_tr_raw[f'Cluster_{k}'] = kmeans.predict(X_tr_phys_sc)
            X_va_raw[f'Cluster_{k}'] = kmeans.predict(X_va_phys_sc)
            X_te_raw[f'Cluster_{k}'] = kmeans.predict(X_te_phys_sc)
            
        for n in [2, 5]:
            pca = PCA(n_components=n, random_state=seed)
            X_tr_pca = pca.fit_transform(X_tr_phys_sc)
            X_va_pca = pca.transform(X_va_phys_sc)
            X_te_pca = pca.transform(X_te_phys_sc)
            for i in range(n):
                X_tr_raw[f'PCA_{n}_{i}'] = X_tr_pca[:, i]
                X_va_raw[f'PCA_{n}_{i}'] = X_va_pca[:, i]
                X_te_raw[f'PCA_{n}_{i}'] = X_te_pca[:, i]

        # 4. Final Feature Selection
        # base_features + TE + Unsupervised + Interactions
        final_features = [f for f in base_features if f not in te_cols] + \
                         [f'{c}_TE' for c in te_cols] + \
                         [f'Cluster_{k}' for k in [3, 6, 12, 50]] + \
                         [f'PCA_{n}_{i}' for n in [2, 5] for i in range(n)] + \
                         interaction_feats
        
        X_tr = X_tr_raw[final_features]
        X_va = X_va_raw[final_features]
        X_te = X_te_raw[final_features]
        
        # 5. Train
        p = cat_params.copy()
        p['random_seed'] = seed
        model = CatBoostClassifier(**p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, verbose=False)
        oof_cat[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_cat += model.predict_proba(X_te)[:, 1] / (N_FOLDS * N_SEEDS)

        p = lgb_params.copy()
        p['seed'] = seed
        model = lgb.LGBMClassifier(**p)
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_names=['valid'], callbacks=callbacks)
        oof_lgb[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_lgb += model.predict_proba(X_te)[:, 1] / (N_FOLDS * N_SEEDS)

        p = xgb_params.copy()
        p['seed'] = seed
        p['early_stopping_rounds'] = 100
        model = xgb.XGBClassifier(**p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_xgb += model.predict_proba(X_te)[:, 1] / (N_FOLDS * N_SEEDS)

print("\n--- Calculating Ensemble ---")
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
    'model': 'Exp49 Interaction Search',
    'cv_final': float(final_cv),
    'cv_individual': {'cat': float(cv_cat), 'lgb': float(cv_lgb), 'xgb': float(cv_xgb)},
    'weights': [float(w) for w in best_weights]
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved to {EXP_DIR}")
