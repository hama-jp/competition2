"""
Exp43d: TabNet Parameter Tuning

Results:
- baseline (n_d=8): CV=0.72504
- larger (n_d=16): CV=0.81114
- larger2 (n_d=32, n_steps=5): CV=0.82561 ← BEST

Ensemble with CatBoost (0.83630):
- CAT + TabNet logit avg: 0.84442 (+0.00811)
- vs CAT + MLP: 0.84425 (+0.00397)

Best config: n_d=32, n_a=32, n_steps=5, lr=1e-2, batch_size=128
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.special import logit, expit
from catboost import CatBoostClassifier, Pool
import torch
import warnings

warnings.filterwarnings('ignore')

N_FOLDS = 5
SEED = 42

def safe_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return logit(p)

def logit_average(probs_list, weights):
    logits = [safe_logit(p) for p in probs_list]
    weighted_logit = sum(w * l for w, l in zip(weights, logits))
    return expit(weighted_logit)

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
    train = pd.read_csv('../../train.csv')
    test = pd.read_csv('../../test.csv')
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

    cols = ['Id', 'Drafted', 'is_train'] + SELECTED_FEATURES
    existing_cols = [c for c in cols if c in data.columns]
    data = data[existing_cols]
    features = [c for c in data.columns if c not in ['Id', 'Drafted', 'is_train']]
    cat_indices = [features.index(c) for c in cat_cols if c in features]
    train_df = data[data['is_train'] == 1].reset_index(drop=True)
    test_df = data[data['is_train'] == 0].reset_index(drop=True)
    target = train_df['Drafted']
    return train_df, test_df, target, features, cat_indices

def train_tabnet_config(X, y, features, config, name):
    print(f"  {name}...", end=" ", flush=True)
    oof = np.zeros(len(X))
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr = X.iloc[train_idx][features].copy()
        X_val = X.iloc[val_idx][features].copy()
        y_tr = y.iloc[train_idx].values
        y_val = y.iloc[val_idx].values

        for col in features:
            med = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(med)
            X_val[col] = X_val[col].fillna(med)

        X_tr = X_tr.values.astype(np.float32)
        X_val = X_val.values.astype(np.float32)

        model = TabNetClassifier(
            n_d=config['n_d'], n_a=config['n_a'], n_steps=config['n_steps'],
            gamma=config.get('gamma', 1.3), lambda_sparse=config.get('lambda_sparse', 1e-3),
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=config['lr']),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax', seed=SEED, verbose=0
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=['auc'],
                  max_epochs=config.get('max_epochs', 200), patience=config.get('patience', 30),
                  batch_size=config['batch_size'], virtual_batch_size=config['batch_size'] // 2, drop_last=False)

        pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = pred
        fold_scores.append(roc_auc_score(y_val, pred))

    cv = roc_auc_score(y, oof)
    std = np.std(fold_scores)
    print(f"CV={cv:.5f} (std={std:.4f})")
    return oof, cv, std

if __name__ == "__main__":
    train_df, test_df, target, features, cat_indices = get_data()
    print(f"Train: {len(train_df)}, Features: {len(features)}\n")

    configs = {
        'baseline': {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'lr': 2e-2, 'batch_size': 256},
        'larger': {'n_d': 16, 'n_a': 16, 'n_steps': 4, 'lr': 1e-2, 'batch_size': 128},
        'larger2': {'n_d': 32, 'n_a': 32, 'n_steps': 5, 'lr': 1e-2, 'batch_size': 128},
        'small_batch': {'n_d': 16, 'n_a': 16, 'n_steps': 4, 'lr': 5e-3, 'batch_size': 64},
        'regularized': {'n_d': 16, 'n_a': 16, 'n_steps': 3, 'lr': 1e-2, 'batch_size': 128, 'lambda_sparse': 1e-2, 'gamma': 1.5},
        'longer': {'n_d': 16, 'n_a': 16, 'n_steps': 4, 'lr': 5e-3, 'batch_size': 128, 'max_epochs': 300, 'patience': 50},
        'minimal': {'n_d': 8, 'n_a': 8, 'n_steps': 2, 'lr': 5e-3, 'batch_size': 64, 'lambda_sparse': 1e-2},
    }

    print("=" * 60)
    print("TabNet Parameter Search")
    print("=" * 60)
    results = {}
    for name, config in configs.items():
        oof, cv, std = train_tabnet_config(train_df, target, features, config, name)
        results[name] = {'oof': oof, 'cv': cv, 'std': std, 'config': config}

    print("\n" + "=" * 60)
    print("RESULTS (sorted by CV)")
    print("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['cv'])
    print(f"{'Config':<15} {'CV':>10} {'Std':>10}")
    print("-" * 35)
    for name, r in sorted_results:
        print(f"{name:<15} {r['cv']:>10.5f} {r['std']:>10.4f}")

    best_name = sorted_results[0][0]
    best_oof = results[best_name]['oof']
    best_cv = results[best_name]['cv']
    print(f"\nBest config: {best_name} (CV={best_cv:.5f})")

    print("\nTraining CatBoost for comparison...")
    oof_cat = np.zeros(len(train_df))
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, target)):
        X_tr, y_tr = train_df.iloc[train_idx][features], target.iloc[train_idx]
        X_val, y_val = train_df.iloc[val_idx][features], target.iloc[val_idx]
        model = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', verbose=False,
                                   allow_writing_files=False, iterations=5000, learning_rate=0.05,
                                   depth=6, random_seed=SEED)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_val, y_val, cat_features=cat_indices)
        model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
        oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]

    cv_cat = roc_auc_score(target, oof_cat)
    print(f"CatBoost CV: {cv_cat:.5f}")

    print("\n--- Ensemble (CAT + Best TabNet) ---")
    corr = np.corrcoef(oof_cat, best_oof)[0, 1]
    print(f"Correlation: {corr:.4f}")

    best_logit_score = 0
    best_w = 0
    for w_cat in np.arange(0.5, 1.0, 0.05):
        logit_ens = logit_average([oof_cat, best_oof], [w_cat, 1-w_cat])
        score = roc_auc_score(target, logit_ens)
        if score > best_logit_score:
            best_logit_score = score
            best_w = w_cat

    print(f"CAT only:  {cv_cat:.5f}")
    print(f"Logit avg: {best_logit_score:.5f} (CAT={best_w:.2f})")
    print(f"Improvement: {best_logit_score - cv_cat:+.5f}")
