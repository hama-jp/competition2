"""
Exp36: CatBoost only with pseudo-labels
- Base: exp21 (31 features)
- Single CatBoost model
- Pseudo-labeling with high-confidence predictions
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

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]
PSEUDO_THRESHOLD = 100  # Top N confident samples

EXP_DIR = '/home/user/competition2/experiments/exp36_catboost_pseudo'
BASE_DIR = '/home/user/competition2'

os.makedirs(EXP_DIR, exist_ok=True)

print("=" * 60)
print("Exp36: CatBoost only with pseudo-labels")
print("=" * 60)

# Load exp13 results
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']

# 31 features (same as exp21)
features = top30_features + ['Agility_3cone_Pos_Diff']
print(f"Features: {len(features)}")

# Load exp07 params
with open('/home/user/competition2/experiments/exp07_final/results.json', 'r') as f:
    exp07_results = json.load(f)
cat_params = exp07_results['best_params_cat']

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
# Feature Engineering
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
print(f"Categorical features: {[features[i] for i in cat_indices]}")

# ==========================================
# Stage 1: Get initial predictions for pseudo-labeling
# ==========================================
print("\n--- Stage 1: Initial predictions ---")

oof_stage1 = np.zeros(len(train_df))
pred_stage1 = np.zeros(len(test_df))

for seed_idx, seed in enumerate(SEEDS):
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        cat_p = cat_params.copy()
        cat_p['random_seed'] = seed
        model_cat = CatBoostClassifier(**cat_p)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
        oof_stage1[va_idx] += model_cat.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_stage1 += model_cat.predict_proba(X_test)[:, 1] / (N_FOLDS * N_SEEDS)

cv_stage1 = roc_auc_score(y_train, oof_stage1)
print(f"Stage 1 CV: {cv_stage1:.5f}")

# ==========================================
# Select pseudo-labels
# ==========================================
confidence = np.abs(pred_stage1 - 0.5)  # Distance from 0.5
top_indices = np.argsort(confidence)[-PSEUDO_THRESHOLD:]

pseudo_labels = (pred_stage1[top_indices] > 0.5).astype(int)
pseudo_X = X_test.iloc[top_indices].reset_index(drop=True)
pseudo_y = pd.Series(pseudo_labels)

print(f"\nPseudo-labels: {len(pseudo_labels)} samples")
print(f"  Positive: {pseudo_labels.sum()}, Negative: {len(pseudo_labels) - pseudo_labels.sum()}")

# ==========================================
# Stage 2: Train with pseudo-labels
# ==========================================
print("\n--- Stage 2: Training with pseudo-labels ---")

X_train_aug = pd.concat([X_train, pseudo_X], ignore_index=True)
y_train_aug = pd.concat([y_train, pseudo_y], ignore_index=True)

oof_stage2 = np.zeros(len(train_df))
pred_stage2 = np.zeros(len(test_df))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ({seed_idx+1}/{N_SEEDS}) ---")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    # For augmented data, we only evaluate on original train indices
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        # Training uses augmented data
        aug_tr_idx = list(tr_idx) + list(range(len(X_train), len(X_train_aug)))
        X_tr = X_train_aug.iloc[aug_tr_idx]
        y_tr = y_train_aug.iloc[aug_tr_idx]

        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        cat_p = cat_params.copy()
        cat_p['random_seed'] = seed
        model_cat = CatBoostClassifier(**cat_p)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
        oof_stage2[va_idx] += model_cat.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_stage2 += model_cat.predict_proba(X_test)[:, 1] / (N_FOLDS * N_SEEDS)

    cv_current = roc_auc_score(y_train, oof_stage2 * N_SEEDS / (seed_idx + 1))
    print(f"  CAT CV: {cv_current:.5f}")

# Final score
cv_final = roc_auc_score(y_train, oof_stage2)

print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"Stage 1 CV (no pseudo): {cv_stage1:.5f}")
print(f"Stage 2 CV (with pseudo): {cv_final:.5f}")
print(f"\nexp33 (no pseudo): CV = 0.85083, LB = 0.85130")
print(f"Difference vs exp33: {cv_final - 0.85083:+.5f}")

# Save submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': pred_stage2
})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

# Save results
results = {
    'n_features': len(features),
    'model': 'CatBoost only with pseudo-labels',
    'pseudo_threshold': PSEUDO_THRESHOLD,
    'cv_stage1': float(cv_stage1),
    'cv_stage2': float(cv_final),
    'exp33_cv': 0.85083,
    'exp33_lb': 0.85130
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSubmission saved to {EXP_DIR}/submission.csv")
