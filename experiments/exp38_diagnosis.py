"""
Quick diagnosis for exp38 CV stability
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/home/user/competition2'
N_FOLDS = 5
SEEDS = [42, 2023, 101]

# exp38 params
EXP38_PARAMS = {
    'learning_rate': 0.07727780074568463,
    'depth': 2,
    'l2_leaf_reg': 0.01673808578875214,
    'subsample': 0.7778987721304084,
    'min_data_in_leaf': 64,
    'random_strength': 1.0930894746349535,
    'iterations': 10000,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'allow_writing_files': False,
}

# Target Encoding
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

print("Loading data...")
train_df, test_df, target, cat_cols = get_data()

with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']
features = top30_features + ['Agility_3cone_Pos_Diff']

X_train = train_df[features]
y_train = target
cat_indices = [features.index(c) for c in cat_cols if c in features]

print(f"Train: {len(train_df)}, Features: {len(features)}")

# Diagnosis
print("\n" + "=" * 60)
print("Diagnosing: exp38_catboost_finetune")
print("=" * 60)

train_aucs = []
val_aucs = []
train_val_gaps = []
early_stop_iters = []
seed_cvs = []

for seed in SEEDS:
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        p = EXP38_PARAMS.copy()
        p['random_seed'] = seed
        model = CatBoostClassifier(**p)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

        train_pred = model.predict_proba(X_tr)[:, 1]
        train_auc = roc_auc_score(y_tr, train_pred)
        val_pred = model.predict_proba(X_va)[:, 1]
        val_auc = roc_auc_score(y_va, val_pred)
        oof[va_idx] = val_pred
        best_iter = model.get_best_iteration()

        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        train_val_gaps.append(train_auc - val_auc)
        early_stop_iters.append(best_iter)

    seed_cv = roc_auc_score(y_train, oof)
    seed_cvs.append(seed_cv)
    print(f"Seed {seed}: CV = {seed_cv:.5f}")

# Summary
mean_train = np.mean(train_aucs)
mean_val = np.mean(val_aucs)
mean_gap = np.mean(train_val_gaps)
std_gap = np.std(train_val_gaps)
mean_iter = np.mean(early_stop_iters)
std_iter = np.std(early_stop_iters)
cv_std = np.std(seed_cvs)

print("\n" + "=" * 60)
print("exp38 DIAGNOSIS RESULTS")
print("=" * 60)

print(f"\n--- Train vs Val Gap ---")
print(f"  Mean Train AUC: {mean_train:.5f}")
print(f"  Mean Val AUC:   {mean_val:.5f}")
print(f"  Mean Gap:       {mean_gap:.5f} (±{std_gap:.5f})")

print(f"\n--- Early Stopping ---")
print(f"  Mean Iterations: {mean_iter:.1f} (±{std_iter:.1f})")

print(f"\n--- CV Stability (KEY METRIC) ---")
print(f"  Seed CVs: {[f'{cv:.5f}' for cv in seed_cvs]}")
print(f"  CV Std:   {cv_std:.5f}")

print("\n" + "=" * 60)
print("COMPARISON WITH KNOWN EXPERIMENTS")
print("=" * 60)
print(f"\n{'Experiment':<30} {'CV Std':<12} {'Iter Std':<12} {'LB':<12} {'Prediction'}")
print("-" * 80)
print(f"{'exp33 (default)':<30} {'0.00249':<12} {'80.3':<12} {'0.85130':<12} {'Best LB'}")
print(f"{'exp37 (wide tune)':<30} {'0.00502':<12} {'148.0':<12} {'0.84655':<12} {'Overfitted'}")
print(f"{'exp38 (fine-tune)':<30} {cv_std:<12.5f} {std_iter:<12.1f} {'?':<12}", end="")

# Prediction
if cv_std < 0.003:
    print("Good LB expected!")
elif cv_std < 0.004:
    print("Moderate LB expected")
else:
    print("Risk of overfitting")

# Save results
results = {
    'exp38_cv_std': float(cv_std),
    'exp38_iter_std': float(std_iter),
    'exp38_mean_gap': float(mean_gap),
    'exp38_seed_cvs': [float(x) for x in seed_cvs],
    'exp33_cv_std': 0.00249,
    'exp37_cv_std': 0.00502,
}
print(f"\nexp38 CV Std: {cv_std:.5f}")
print(f"exp33 CV Std: 0.00249 (LB=0.85130)")
print(f"exp37 CV Std: 0.00502 (LB=0.84655)")
