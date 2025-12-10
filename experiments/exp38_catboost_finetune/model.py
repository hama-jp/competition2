"""
Exp38: CatBoost fine-tuning around exp33 parameters
- Base: exp33 (LB 0.85130)
- Strategy: Narrow search range around exp33's best params
- Goal: Small improvements without overfitting
"""
import pandas as pd
import numpy as np
import os
import json
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_FOLDS = 5
N_TUNING_SEEDS = 3
N_FINAL_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]
N_TRIALS = 30  # Fewer trials, narrower range

EXP_DIR = '/home/user/competition2/experiments/exp38_catboost_finetune'
BASE_DIR = '/home/user/competition2'

os.makedirs(EXP_DIR, exist_ok=True)

print("=" * 60)
print("Exp38: CatBoost fine-tuning around exp33 parameters")
print("=" * 60)

# exp33/exp07 base parameters
BASE_PARAMS = {
    'learning_rate': 0.07718772443488796,
    'depth': 3,
    'l2_leaf_reg': 0.0033458292447738312,
    'subsample': 0.8523245279212943,
    'min_data_in_leaf': 71,
    'random_strength': 1.2032200146196355,
}

print(f"\nBase params (exp33):")
for k, v in BASE_PARAMS.items():
    print(f"  {k}: {v}")

# Load exp13 results
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']

# 31 features (same as exp21/exp33)
features = top30_features + ['Agility_3cone_Pos_Diff']
print(f"\nFeatures: {len(features)}")

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
# Feature Engineering (same as exp33)
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
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ==========================================
# Optuna fine-tuning (narrow range)
# ==========================================
print("\n" + "=" * 60)
print("Fine-tuning with Optuna (narrow range around exp33)")
print("=" * 60)

def objective(trial):
    # Narrow search range: +/- 20% around base params
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.06, 0.10),  # base: 0.077
        'depth': trial.suggest_int('depth', 2, 4),  # base: 3
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.001, 0.1, log=True),  # base: 0.0033
        'subsample': trial.suggest_float('subsample', 0.75, 0.95),  # base: 0.85
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 100),  # base: 71
        'random_strength': trial.suggest_float('random_strength', 0.8, 1.6),  # base: 1.2
        'iterations': 10000,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': False,
        'allow_writing_files': False,
    }

    scores = []
    for seed in SEEDS[:N_TUNING_SEEDS]:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X_train))

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

            params['random_seed'] = seed
            model = CatBoostClassifier(**params)
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
            oof[va_idx] = model.predict_proba(X_va)[:, 1]

        scores.append(roc_auc_score(y_train, oof))

    return np.mean(scores)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

# Add exp33 params as first trial
study.enqueue_trial({
    'learning_rate': 0.077,
    'depth': 3,
    'l2_leaf_reg': 0.0033,
    'subsample': 0.85,
    'min_data_in_leaf': 71,
    'random_strength': 1.2,
})

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest trial: {study.best_trial.number}")
print(f"Best CV (tuning): {study.best_value:.6f}")
print(f"\nBest params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Compare with base
print(f"\nDifference from base params:")
for k, v in study.best_params.items():
    if k in BASE_PARAMS:
        diff = v - BASE_PARAMS[k]
        pct = (diff / BASE_PARAMS[k]) * 100 if BASE_PARAMS[k] != 0 else 0
        print(f"  {k}: {diff:+.6f} ({pct:+.1f}%)")

# ==========================================
# Final training with best params
# ==========================================
print("\n" + "=" * 60)
print("Final training with best params (5 seeds)")
print("=" * 60)

best_params = study.best_params.copy()
best_params['iterations'] = 10000
best_params['loss_function'] = 'Logloss'
best_params['eval_metric'] = 'AUC'
best_params['verbose'] = False
best_params['allow_writing_files'] = False

oof_final = np.zeros(len(train_df))
pred_final = np.zeros(len(test_df))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ({seed_idx+1}/{N_FINAL_SEEDS}) ---")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        best_params['random_seed'] = seed
        model = CatBoostClassifier(**best_params)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

        oof_final[va_idx] += model.predict_proba(X_va)[:, 1] / N_FINAL_SEEDS
        pred_final += model.predict_proba(X_test)[:, 1] / (N_FOLDS * N_FINAL_SEEDS)

    cv_now = roc_auc_score(y_train, oof_final * N_FINAL_SEEDS / (seed_idx + 1))
    print(f"  CV: {cv_now:.5f}")

cv_final = roc_auc_score(y_train, oof_final)

print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"Fine-tuned CatBoost CV: {cv_final:.5f}")
print(f"\nexp33 (base params): CV = 0.85083, LB = 0.85130")
print(f"exp37 (wide tuning): CV = 0.85307, LB = 0.84655")
print(f"Difference vs exp33: {cv_final - 0.85083:+.5f}")

# Save submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': pred_final
})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

# Save results
results = {
    'n_features': len(features),
    'model': 'CatBoost fine-tuned',
    'n_trials': N_TRIALS,
    'base_params': BASE_PARAMS,
    'best_params': best_params,
    'tuning_cv': study.best_value,
    'final_cv': float(cv_final),
    'exp33_cv': 0.85083,
    'exp33_lb': 0.85130,
    'exp37_cv': 0.85307,
    'exp37_lb': 0.84655,
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSubmission saved to {EXP_DIR}/submission.csv")
