"""
Exp09: TabNet Improvement
- StandardScaler for numerical features
- Optuna hyperparameter optimization
- More epochs and better architecture
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import optuna
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 3
SEEDS = [42, 2023, 101]
OPTUNA_TRIALS = 30

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
    print("Loading and preprocessing...")
    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

    train['is_train'] = 1
    test['is_train'] = 0
    test['Drafted'] = np.nan
    data = pd.concat([train, test], sort=False).reset_index(drop=True)

    # Feature Engineering (same as exp07)
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

    exclude_cols = ['Id', 'Drafted', 'is_train']
    features = [c for c in train_df.columns if c not in exclude_cols]

    return train_df, test_df, target, features

def preprocess_for_tabnet(X_train, X_test, features):
    """Preprocess data for TabNet with scaling and NaN handling"""
    X_train_processed = X_train[features].copy()
    X_test_processed = X_test[features].copy()

    # Fill NaN with median
    for col in features:
        if X_train_processed[col].isna().any():
            median_val = X_train_processed[col].median()
            X_train_processed[col] = X_train_processed[col].fillna(median_val)
            X_test_processed[col] = X_test_processed[col].fillna(median_val)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)

def optimize_tabnet(X_scaled, y):
    """Optimize TabNet hyperparameters with Optuna"""
    print("Optimizing TabNet hyperparameters...")

    def objective(trial):
        params = {
            'n_d': trial.suggest_int('n_d', 8, 64),
            'n_a': trial.suggest_int('n_a', 8, 64),
            'n_steps': trial.suggest_int('n_steps', 3, 10),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'n_independent': trial.suggest_int('n_independent', 1, 5),
            'n_shared': trial.suggest_int('n_shared', 1, 5),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-2, log=True),
            'momentum': trial.suggest_float('momentum', 0.01, 0.4),
            'lr': trial.suggest_float('lr', 1e-3, 0.1, log=True),
        }

        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for tr_idx, va_idx in kf.split(X_scaled, y):
            X_tr, y_tr = X_scaled[tr_idx], y.iloc[tr_idx].values
            X_va, y_va = X_scaled[va_idx], y.iloc[va_idx].values

            model = TabNetClassifier(
                n_d=params['n_d'],
                n_a=params['n_a'],
                n_steps=params['n_steps'],
                gamma=params['gamma'],
                n_independent=params['n_independent'],
                n_shared=params['n_shared'],
                lambda_sparse=params['lambda_sparse'],
                momentum=params['momentum'],
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=params['lr']),
                scheduler_params={"step_size": 15, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                seed=42,
                verbose=0
            )

            model.fit(
                X_train=X_tr, y_train=y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric=['auc'],
                max_epochs=150,
                patience=20,
                batch_size=256,
                virtual_batch_size=128,
                drop_last=False
            )

            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print(f"  Best TabNet CV: {study.best_value:.5f}")
    return study.best_params

def train_tabnet(X_train_scaled, X_test_scaled, y, params):
    """Train TabNet with optimized parameters"""
    print("Training TabNet with optimized params...")

    oof = np.zeros(len(X_train_scaled))
    preds = np.zeros(len(X_test_scaled))

    for seed in SEEDS:
        print(f"  Seed {seed}...")
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_scaled, y)):
            X_tr, y_tr = X_train_scaled[tr_idx], y.iloc[tr_idx].values
            X_va, y_va = X_train_scaled[va_idx], y.iloc[va_idx].values

            model = TabNetClassifier(
                n_d=params['n_d'],
                n_a=params['n_a'],
                n_steps=params['n_steps'],
                gamma=params['gamma'],
                n_independent=params['n_independent'],
                n_shared=params['n_shared'],
                lambda_sparse=params['lambda_sparse'],
                momentum=params['momentum'],
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=params['lr']),
                scheduler_params={"step_size": 15, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                seed=seed,
                verbose=0
            )

            model.fit(
                X_train=X_tr, y_train=y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric=['auc'],
                max_epochs=200,
                patience=30,
                batch_size=256,
                virtual_batch_size=128,
                drop_last=False
            )

            oof[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
            preds += model.predict_proba(X_test_scaled)[:, 1] / (N_FOLDS * len(SEEDS))

    cv = roc_auc_score(y, oof)
    print(f"  TabNet CV: {cv:.5f}")
    return oof, preds, cv

if __name__ == '__main__':
    print("="*50)
    print("Exp09: TabNet Improvement")
    print("="*50)

    train_df, test_df, target, features = get_data()
    print(f"Features: {len(features)}")

    # Preprocess for TabNet
    X_train_scaled, X_test_scaled = preprocess_for_tabnet(train_df, test_df, features)

    # Optimize and train
    best_params = optimize_tabnet(X_train_scaled, target)
    oof_tabnet, pred_tabnet, cv_tabnet = train_tabnet(X_train_scaled, X_test_scaled, target, best_params)

    print(f"\n{'='*50}")
    print(f"TabNet CV: {cv_tabnet:.5f}")
    print(f"Best params: {best_params}")
    print(f"{'='*50}")

    # Save results
    submission = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))
    submission['Drafted'] = pred_tabnet
    submission.to_csv(os.path.join(EXP_DIR, 'submission.csv'), index=False)

    results = {
        'experiment': 'exp09_tabnet_improved',
        'timestamp': datetime.now().isoformat(),
        'cv_tabnet': cv_tabnet,
        'best_params': best_params,
        'n_features': len(features)
    }

    with open(os.path.join(EXP_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    np.save(os.path.join(EXP_DIR, 'oof_tabnet.npy'), oof_tabnet)
    np.save(os.path.join(EXP_DIR, 'pred_tabnet.npy'), pred_tabnet)

    print(f"\nResults saved to {EXP_DIR}")
