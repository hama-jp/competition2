"""
Exp43c: TabNet + CatBoost Ensemble with Logit Average

TabNet: Attentionベースのテーブルデータ専用NN
- GBDTに近い動作（特徴量選択をAttentionで学習）
- 解釈可能性が高い
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.special import logit, expit
from catboost import CatBoostClassifier, Pool
import torch
import warnings

warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 3  # TabNetは遅いので3に
SEEDS = [42, 2023, 101]

def safe_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return logit(p)

def logit_average(probs_list, weights):
    logits = [safe_logit(p) for p in probs_list]
    weighted_logit = sum(w * l for w, l in zip(weights, logits))
    return expit(weighted_logit)

# Data
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

def train_tabnet(X, y, X_test, features):
    print("Training TabNet...")
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ")
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr = X.iloc[train_idx][features].copy()
            X_val = X.iloc[val_idx][features].copy()
            X_te = X_test[features].copy()
            y_tr = y.iloc[train_idx].values
            y_val = y.iloc[val_idx].values

            # Fill NaN
            for col in features:
                med = X_tr[col].median()
                X_tr[col] = X_tr[col].fillna(med)
                X_val[col] = X_val[col].fillna(med)
                X_te[col] = X_te[col].fillna(med)

            X_tr = X_tr.values.astype(np.float32)
            X_val = X_val.values.astype(np.float32)
            X_te = X_te.values.astype(np.float32)

            model = TabNetClassifier(
                n_d=8, n_a=8,  # 小さめに
                n_steps=3,
                gamma=1.3,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size": 50, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type='entmax',
                seed=seed,
                verbose=0
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=['auc'],
                max_epochs=100,
                patience=20,
                batch_size=256,
                virtual_batch_size=128,
                drop_last=False
            )

            oof[val_idx] += model.predict_proba(X_val)[:, 1] / N_SEEDS
            preds += model.predict_proba(X_te)[:, 1] / (N_FOLDS * N_SEEDS)

        print(f"done")

    cv = roc_auc_score(y, oof)
    print(f"TabNet CV: {cv:.5f}")
    return oof, preds, cv

def train_cat(X, y, X_test, features, cat_indices):
    print("Training CatBoost...")
    params = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC',
        'verbose': False, 'allow_writing_files': False,
        'iterations': 5000, 'learning_rate': 0.05, 'depth': 6
    }
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        params['random_seed'] = seed
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx][features], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx][features], y.iloc[val_idx]

            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_val, y_val, cat_features=cat_indices)
            model = CatBoostClassifier(**params)
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

            oof[val_idx] += model.predict_proba(X_val)[:, 1] / N_SEEDS
            preds += model.predict_proba(X_test[features])[:, 1] / (N_FOLDS * N_SEEDS)

    cv = roc_auc_score(y, oof)
    print(f"CAT CV: {cv:.5f}")
    return oof, preds, cv

if __name__ == "__main__":
    train_df, test_df, target, features, cat_indices = get_data()
    print(f"Train: {len(train_df)}, Features: {len(features)}\n")

    # Train models
    oof_cat, pred_cat, cv_cat = train_cat(train_df, target, test_df, features, cat_indices)
    oof_tabnet, pred_tabnet, cv_tabnet = train_tabnet(train_df, target, test_df, features)

    # Results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"CatBoost: CV={cv_cat:.5f}")
    print(f"TabNet:   CV={cv_tabnet:.5f}")

    # Distribution
    print("\n--- Distribution ---")
    print(f"CAT:    mean={np.mean(oof_cat):.4f}, >0.9={np.mean(oof_cat>0.9)*100:.1f}%")
    print(f"TabNet: mean={np.mean(oof_tabnet):.4f}, >0.9={np.mean(oof_tabnet>0.9)*100:.1f}%")
    print(f"Correlation: {np.corrcoef(oof_cat, oof_tabnet)[0,1]:.4f}")

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

    print(f"CAT only:    CV={cv_cat:.5f}")
    print(f"Simple avg:  CV={best_simple['score']:.5f} (CAT={best_simple['w']:.2f})")
    print(f"Logit avg:   CV={best_logit['score']:.5f} (CAT={best_logit['w']:.2f})")
    print(f"Improvement: {best_logit['score'] - cv_cat:+.5f}")

    # Save submissions
    submission = pd.read_csv('../../sample_submission.csv')

    submission['Drafted'] = pred_cat
    submission.to_csv('submission_cat_only.csv', index=False)

    submission['Drafted'] = pred_tabnet
    submission.to_csv('submission_tabnet_only.csv', index=False)

    w = best_logit['w']
    logit_pred = logit_average([pred_cat, pred_tabnet], [w, 1-w])
    submission['Drafted'] = logit_pred
    submission.to_csv('submission_cat_tabnet_logit.csv', index=False)

    print("\nSubmissions saved!")
