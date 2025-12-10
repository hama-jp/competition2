"""
Exp42: Post-processing on exp33 predictions
- Base: exp33 (best LB = 0.85130)
- Try various post-processing techniques
- Note: AUC is rank-based, so only rank-changing transformations can help
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

EXP_DIR = '/home/user/competition2/experiments/exp42_postprocessing'
BASE_DIR = '/home/user/competition2'

# exp33 params
CAT_PARAMS = {
    'learning_rate': 0.07718772443488796,
    'depth': 3,
    'l2_leaf_reg': 0.0033458292447738312,
    'subsample': 0.8523245279212943,
    'min_data_in_leaf': 71,
    'random_strength': 1.2032200146196355,
    'iterations': 10000,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'allow_writing_files': False,
}

print("=" * 60)
print("Exp42: Post-processing Analysis")
print("=" * 60)

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

# Load data
print("\nLoading data...")
train_df, test_df, target, cat_cols = get_data()

with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']
features = top30_features + ['Agility_3cone_Pos_Diff']

X_train = train_df[features]
y_train = target
X_test = test_df[features]
cat_indices = [features.index(c) for c in cat_cols if c in features]

# ==========================================
# Step 1: Generate OOF and test predictions
# ==========================================
print("\n" + "=" * 60)
print("Step 1: Generate base predictions (exp33 style)")
print("=" * 60)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for seed_idx, seed in enumerate(SEEDS):
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        p = CAT_PARAMS.copy()
        p['random_seed'] = seed
        model = CatBoostClassifier(**p)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

        oof_preds[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        test_preds += model.predict_proba(X_test)[:, 1] / (N_FOLDS * N_SEEDS)

base_cv = roc_auc_score(y_train, oof_preds)
print(f"Base CV: {base_cv:.5f}")

# ==========================================
# Step 2: Analyze OOF predictions
# ==========================================
print("\n" + "=" * 60)
print("Step 2: Analyze OOF predictions")
print("=" * 60)

print(f"\nOOF prediction distribution:")
print(f"  Mean: {oof_preds.mean():.4f}")
print(f"  Std:  {oof_preds.std():.4f}")
print(f"  Min:  {oof_preds.min():.4f}")
print(f"  Max:  {oof_preds.max():.4f}")

print(f"\nTest prediction distribution:")
print(f"  Mean: {test_preds.mean():.4f}")
print(f"  Std:  {test_preds.std():.4f}")
print(f"  Min:  {test_preds.min():.4f}")
print(f"  Max:  {test_preds.max():.4f}")

# Check calibration
print(f"\nCalibration check:")
print(f"  Target mean: {y_train.mean():.4f}")
print(f"  OOF pred mean: {oof_preds.mean():.4f}")
print(f"  Test pred mean: {test_preds.mean():.4f}")

# ==========================================
# Step 3: Try post-processing methods
# ==========================================
print("\n" + "=" * 60)
print("Step 3: Post-processing methods")
print("=" * 60)

results = {}

# Method 1: Raw (baseline)
results['raw'] = {
    'cv': base_cv,
    'test_preds': test_preds.copy(),
}
print(f"\n1. Raw (baseline): CV = {base_cv:.5f}")

# Method 2: Isotonic Calibration
print("\n2. Isotonic Calibration...")
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(oof_preds, y_train)
oof_iso = iso_reg.transform(oof_preds)
test_iso = iso_reg.transform(test_preds)
cv_iso = roc_auc_score(y_train, oof_iso)
results['isotonic'] = {
    'cv': cv_iso,
    'test_preds': test_iso,
}
print(f"   CV = {cv_iso:.5f} (diff: {cv_iso - base_cv:+.5f})")

# Method 3: Power transformation (try different powers)
print("\n3. Power transformations...")
for power in [0.5, 0.8, 1.2, 1.5, 2.0]:
    # Apply power while preserving relative order for values in (0,1)
    oof_pow = oof_preds ** power
    test_pow = test_preds ** power
    cv_pow = roc_auc_score(y_train, oof_pow)
    results[f'power_{power}'] = {
        'cv': cv_pow,
        'test_preds': test_pow,
    }
    diff = cv_pow - base_cv
    print(f"   Power {power}: CV = {cv_pow:.5f} (diff: {diff:+.5f})")

# Method 4: Rank-based transformation (percentile)
print("\n4. Rank transformation...")
from scipy.stats import rankdata
oof_rank = rankdata(oof_preds) / len(oof_preds)
test_rank = rankdata(test_preds) / len(test_preds)
cv_rank = roc_auc_score(y_train, oof_rank)
results['rank'] = {
    'cv': cv_rank,
    'test_preds': test_rank,
}
print(f"   CV = {cv_rank:.5f} (same as raw for AUC)")

# Method 5: Clip extreme values
print("\n5. Clipping extreme values...")
for clip_val in [0.01, 0.05, 0.1]:
    oof_clip = np.clip(oof_preds, clip_val, 1 - clip_val)
    test_clip = np.clip(test_preds, clip_val, 1 - clip_val)
    cv_clip = roc_auc_score(y_train, oof_clip)
    results[f'clip_{clip_val}'] = {
        'cv': cv_clip,
        'test_preds': test_clip,
    }
    print(f"   Clip [{clip_val}, {1-clip_val}]: CV = {cv_clip:.5f}")

# Method 6: Adjust to match target distribution
print("\n6. Distribution matching...")
target_mean = y_train.mean()
oof_adjusted = oof_preds - oof_preds.mean() + target_mean
test_adjusted = test_preds - test_preds.mean() + target_mean
oof_adjusted = np.clip(oof_adjusted, 0.001, 0.999)
test_adjusted = np.clip(test_adjusted, 0.001, 0.999)
cv_adj = roc_auc_score(y_train, oof_adjusted)
results['dist_match'] = {
    'cv': cv_adj,
    'test_preds': test_adjusted,
}
print(f"   CV = {cv_adj:.5f}")

# Method 7: Blend with uniform prior
print("\n7. Blend with prior...")
for alpha in [0.1, 0.2, 0.3]:
    prior = target_mean
    oof_blend = (1 - alpha) * oof_preds + alpha * prior
    test_blend = (1 - alpha) * test_preds + alpha * prior
    cv_blend = roc_auc_score(y_train, oof_blend)
    results[f'blend_prior_{alpha}'] = {
        'cv': cv_blend,
        'test_preds': test_blend,
    }
    print(f"   Alpha {alpha}: CV = {cv_blend:.5f}")

# ==========================================
# Step 4: Summary and best method
# ==========================================
print("\n" + "=" * 60)
print("Step 4: Summary")
print("=" * 60)

print(f"\n{'Method':<25} {'CV':<12} {'Diff vs Raw':<12}")
print("-" * 50)
for name, r in sorted(results.items(), key=lambda x: -x[1]['cv']):
    diff = r['cv'] - base_cv
    print(f"{name:<25} {r['cv']:.5f}      {diff:+.5f}")

# Note: For AUC, most transformations won't change the score
# since AUC is rank-based. Only isotonic might help slightly.

# ==========================================
# Step 5: Save best result
# ==========================================
print("\n" + "=" * 60)
print("Step 5: Save submissions")
print("=" * 60)

# Save raw (exp33 equivalent)
submission_raw = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': test_preds
})
submission_raw.to_csv(f'{EXP_DIR}/submission_raw.csv', index=False)
print(f"Saved: submission_raw.csv (CV={base_cv:.5f})")

# Save isotonic calibrated
submission_iso = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': test_iso
})
submission_iso.to_csv(f'{EXP_DIR}/submission_isotonic.csv', index=False)
print(f"Saved: submission_isotonic.csv (CV={cv_iso:.5f})")

# Find best power transformation
best_power = max([k for k in results.keys() if k.startswith('power_')],
                  key=lambda x: results[x]['cv'])
submission_power = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': results[best_power]['test_preds']
})
submission_power.to_csv(f'{EXP_DIR}/submission_{best_power}.csv', index=False)
print(f"Saved: submission_{best_power}.csv (CV={results[best_power]['cv']:.5f})")

# Results JSON
results_json = {
    'base_cv': float(base_cv),
    'methods': {k: {'cv': float(v['cv'])} for k, v in results.items()},
    'oof_stats': {
        'mean': float(oof_preds.mean()),
        'std': float(oof_preds.std()),
    },
    'test_stats': {
        'mean': float(test_preds.mean()),
        'std': float(test_preds.std()),
    },
    'exp33_lb': 0.85130,
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\n注意: AUCはランクベースの指標のため、")
print(f"ほとんどの変換でCVは変わりません。")
print(f"Isotonic calibrationのみ、わずかに変化する可能性があります。")
