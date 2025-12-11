"""
Exp43e: TabNet + CatBoost (exp33条件で公平比較)

exp33の条件:
- 31特徴量 + Target Encoding
- N_SEEDS = 5, N_FOLDS = 5
- exp07のCatBoostパラメータ

TabNet: larger2設定 (n_d=32, n_a=32, n_steps=5)
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier, Pool
from scipy.special import logit, expit
import torch
import warnings

warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

BASE_DIR = '/home/user/competition2'

def safe_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return logit(p)

def logit_average(probs_list, weights):
    logits = [safe_logit(p) for p in probs_list]
    weighted_logit = sum(w * l for w, l in zip(weights, logits))
    return expit(weighted_logit)

# exp13の特徴量を読み込み
with open(f'{BASE_DIR}/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']
features = top30_features + ['Agility_3cone_Pos_Diff']  # 31 features

# exp07のCatBoostパラメータ
with open(f'{BASE_DIR}/experiments/exp07_final/results.json', 'r') as f:
    exp07_results = json.load(f)
cat_params = exp07_results['best_params_cat']

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

    # Target Encoding
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

print("=" * 60)
print("Exp43e: TabNet + CatBoost (exp33条件)")
print("=" * 60)

train_df, test_df, target, cat_cols = get_data()
X_train = train_df[features]
X_test = test_df[features]
cat_indices = [features.index(c) for c in cat_cols if c in features]

print(f"Features: {len(features)}")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ==========================================
# CatBoost (exp33と同じ)
# ==========================================
print("\n--- CatBoost Training (5 seeds) ---")
oof_cat = np.zeros(len(train_df))
pred_cat = np.zeros(len(test_df))

for seed_idx, seed in enumerate(SEEDS):
    print(f"Seed {seed}...", end=" ", flush=True)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, target)):
        X_tr, y_tr = X_train.iloc[tr_idx], target.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], target.iloc[va_idx]

        cat_p = cat_params.copy()
        cat_p['random_seed'] = seed
        model = CatBoostClassifier(**cat_p)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

        oof_cat[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_cat += model.predict_proba(X_test)[:, 1] / (N_FOLDS * N_SEEDS)
    print("done")

cv_cat = roc_auc_score(target, oof_cat)
print(f"CatBoost CV: {cv_cat:.5f}")

# ==========================================
# TabNet (larger2設定, 5 seeds)
# ==========================================
print("\n--- TabNet Training (larger2, 5 seeds) ---")
oof_tabnet = np.zeros(len(train_df))
pred_tabnet = np.zeros(len(test_df))

tabnet_config = {
    'n_d': 32, 'n_a': 32, 'n_steps': 5,
    'lr': 1e-2, 'batch_size': 128
}

for seed_idx, seed in enumerate(SEEDS):
    print(f"Seed {seed}...", end=" ", flush=True)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, target)):
        X_tr = X_train.iloc[tr_idx].copy()
        X_va = X_train.iloc[va_idx].copy()
        X_te = X_test.copy()
        y_tr = target.iloc[tr_idx].values
        y_va = target.iloc[va_idx].values

        # Fill NaN
        for col in features:
            med = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(med)
            X_va[col] = X_va[col].fillna(med)
            X_te[col] = X_te[col].fillna(med)

        X_tr_np = X_tr.values.astype(np.float32)
        X_va_np = X_va.values.astype(np.float32)
        X_te_np = X_te.values.astype(np.float32)

        model = TabNetClassifier(
            n_d=tabnet_config['n_d'],
            n_a=tabnet_config['n_a'],
            n_steps=tabnet_config['n_steps'],
            gamma=1.3,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=tabnet_config['lr']),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            seed=seed,
            verbose=0
        )

        model.fit(
            X_tr_np, y_tr,
            eval_set=[(X_va_np, y_va)],
            eval_metric=['auc'],
            max_epochs=200,
            patience=30,
            batch_size=tabnet_config['batch_size'],
            virtual_batch_size=tabnet_config['batch_size'] // 2,
            drop_last=False
        )

        oof_tabnet[va_idx] += model.predict_proba(X_va_np)[:, 1] / N_SEEDS
        pred_tabnet += model.predict_proba(X_te_np)[:, 1] / (N_FOLDS * N_SEEDS)
    print("done")

cv_tabnet = roc_auc_score(target, oof_tabnet)
print(f"TabNet CV: {cv_tabnet:.5f}")

# ==========================================
# Results
# ==========================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"CatBoost CV: {cv_cat:.5f}")
print(f"TabNet CV:   {cv_tabnet:.5f}")

# Distribution
print("\n--- Distribution ---")
print(f"CAT:    mean={np.mean(oof_cat):.4f}, std={np.std(oof_cat):.4f}, >0.9={np.mean(oof_cat>0.9)*100:.1f}%")
print(f"TabNet: mean={np.mean(oof_tabnet):.4f}, std={np.std(oof_tabnet):.4f}, >0.9={np.mean(oof_tabnet>0.9)*100:.1f}%")

corr = np.corrcoef(oof_cat, oof_tabnet)[0, 1]
print(f"Correlation: {corr:.4f}")

# Ensemble
print("\n--- Ensemble ---")
best_simple = {'score': 0, 'w': 0}
best_logit = {'score': 0, 'w': 0}

for w_cat in np.arange(0.5, 1.0, 0.05):
    w_nn = 1.0 - w_cat
    simple = w_cat * oof_cat + w_nn * oof_tabnet
    logit_ens = logit_average([oof_cat, oof_tabnet], [w_cat, w_nn])

    s = roc_auc_score(target, simple)
    l = roc_auc_score(target, logit_ens)

    if s > best_simple['score']:
        best_simple = {'score': s, 'w': w_cat}
    if l > best_logit['score']:
        best_logit = {'score': l, 'w': w_cat}

print(f"CAT only:    {cv_cat:.5f}")
print(f"Simple avg:  {best_simple['score']:.5f} (CAT={best_simple['w']:.2f})")
print(f"Logit avg:   {best_logit['score']:.5f} (CAT={best_logit['w']:.2f})")
print(f"Improvement: {best_logit['score'] - cv_cat:+.5f}")

# Save submissions
submission = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')

submission['Drafted'] = pred_cat
submission.to_csv('submission_cat_only.csv', index=False)

submission['Drafted'] = pred_tabnet
submission.to_csv('submission_tabnet_only.csv', index=False)

w = best_logit['w']
logit_pred = logit_average([pred_cat, pred_tabnet], [w, 1-w])
submission['Drafted'] = logit_pred
submission.to_csv('submission_logit_avg.csv', index=False)

print("\nSubmissions saved!")
