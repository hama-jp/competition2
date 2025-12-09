"""
Exp30: Non-linear Feature Transformations
- Analyze features with high Chatterjee but low Pearson correlation
- Design appropriate non-linear transformations
- Validate with CV
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

EXP_DIR = '/home/user/competition2/experiments/exp30_nonlinear_features'
BASE_DIR = '/home/user/competition2'

print("=" * 60)
print("Exp30: Non-linear Feature Transformations")
print("=" * 60)

# ==========================================
# Load raw data for analysis
# ==========================================
train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
print(f"\nTraining samples: {len(train)}")

# ==========================================
# Analyze key non-linear features
# ==========================================
print("\n" + "=" * 60)
print("Analyzing Non-linear Features")
print("=" * 60)

# Features to analyze (excluding Age which is done)
nonlinear_candidates = [
    'Jump_Efficiency',  # (Vertical + Broad) / Weight
    'Height',
    'Agility_3cone',
    'Agility_Sum',      # 3cone + Shuttle
    'BMI',
]

# Compute derived features for analysis
train['BMI'] = train['Weight'] / (train['Height'] ** 2)
train['Jump_Efficiency'] = (train['Vertical_Jump'] + train['Broad_Jump']) / train['Weight']
train['Agility_Sum'] = train['Agility_3cone'] + train['Shuttle']

def analyze_feature_bins(data, feature, target_col='Drafted', n_bins=10):
    """Analyze draft rate by feature bins"""
    valid = data[[feature, target_col]].dropna()
    if len(valid) < 100:
        return None

    # Create bins
    try:
        valid['bin'] = pd.qcut(valid[feature], n_bins, labels=False, duplicates='drop')
    except ValueError:
        valid['bin'] = pd.cut(valid[feature], n_bins, labels=False)

    stats_df = valid.groupby('bin').agg({
        feature: ['mean', 'count'],
        target_col: 'mean'
    }).round(3)
    stats_df.columns = ['feat_mean', 'count', 'draft_rate']
    return stats_df

print("\n--- Draft Rate by Feature Bins ---")
for feat in nonlinear_candidates:
    if feat in train.columns:
        print(f"\n{feat}:")
        stats_df = analyze_feature_bins(train, feat)
        if stats_df is not None:
            print(f"  {'Bin':>4} {'Mean':>10} {'Count':>8} {'Draft Rate':>12}")
            print("  " + "-" * 40)
            for idx, row in stats_df.iterrows():
                print(f"  {idx:4} {row['feat_mean']:10.3f} {int(row['count']):8} {row['draft_rate']:12.3f}")

            # Find optimal range
            max_rate = stats_df['draft_rate'].max()
            min_rate = stats_df['draft_rate'].min()
            optimal_bin = stats_df['draft_rate'].idxmax()
            print(f"  Optimal bin: {optimal_bin} (rate: {max_rate:.3f})")
            print(f"  Range: {min_rate:.3f} - {max_rate:.3f}")

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
# Data Preparation with new transformations
# ==========================================
def get_data_with_transforms():
    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

    train['is_train'] = 1
    test['is_train'] = 0
    test['Drafted'] = np.nan
    data = pd.concat([train, test], sort=False).reset_index(drop=True)

    measure_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    data['Missing_Count'] = data[measure_cols].isnull().sum(axis=1)

    # Base features
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

    # ==========================================
    # NEW: Non-linear transformations (exp28 + new)
    # ==========================================

    # 1. Age transformations (from exp28)
    data['Age_old_penalty'] = np.maximum(0, data['Age'] - 23)

    # 2. Height transformations
    # Height seems to have middle-optimal pattern
    height_mean = data['Height'].mean()
    data['Height_dist_mean'] = np.abs(data['Height'] - height_mean)
    data['Height_dist_mean_sq'] = (data['Height'] - height_mean) ** 2
    # Tall/short indicators
    data['Height_tall'] = (data['Height'] >= 75).astype(float)  # > 75 inches
    data['Height_short'] = (data['Height'] <= 71).astype(float)  # < 71 inches

    # 3. Agility transformations
    # Agility (lower is better) - fast agility bonus
    agility_q25 = data['Agility_3cone'].quantile(0.25)
    agility_q75 = data['Agility_3cone'].quantile(0.75)
    data['Agility_fast'] = (data['Agility_3cone'] <= agility_q25).astype(float)
    data['Agility_slow'] = (data['Agility_3cone'] >= agility_q75).astype(float)
    data['Agility_penalty'] = np.maximum(0, data['Agility_3cone'] - 7.2)  # Penalty for slow

    # Agility_Sum transformations
    agility_sum_median = data['Agility_Sum'].median()
    data['Agility_Sum_fast'] = (data['Agility_Sum'] <= agility_sum_median).astype(float)
    data['Agility_Sum_penalty'] = np.maximum(0, data['Agility_Sum'] - 11.5)

    # 4. BMI transformations
    # BMI often has optimal range
    bmi_mean = data['BMI'].mean()
    data['BMI_dist_optimal'] = np.abs(data['BMI'] - 0.038)  # ~optimal BMI
    data['BMI_dist_optimal_sq'] = (data['BMI'] - 0.038) ** 2
    data['BMI_high'] = (data['BMI'] >= 0.042).astype(float)
    data['BMI_low'] = (data['BMI'] <= 0.034).astype(float)

    # 5. Jump_Efficiency transformations
    je_median = data['Jump_Efficiency'].median()
    data['JumpEff_high'] = (data['Jump_Efficiency'] >= je_median).astype(float)
    data['JumpEff_bonus'] = np.maximum(0, data['Jump_Efficiency'] - je_median)

    # 6. Combined non-linear features
    # Young + Fast agility = high potential
    data['Young_Fast'] = ((data['Age'] <= 22) & (data['Agility_3cone'] <= agility_q25)).astype(float)
    # Old + Slow = red flag
    data['Old_Slow'] = ((data['Age'] >= 24) & (data['Agility_3cone'] >= agility_q75)).astype(float)

    # Position-normalized stats
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

    return train_df, test_df, target


# ==========================================
# Quick CV function
# ==========================================
def quick_cv(train_df, target, features, n_folds=5, seed=42):
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'n_estimators': 500,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': seed
    }

    oof = np.zeros(len(train_df))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X = train_df[features].values
    y = target.values

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return roc_auc_score(target, oof)


# ==========================================
# Load data and test transformations
# ==========================================
print("\n" + "=" * 60)
print("Loading data with new transformations...")
print("=" * 60)

train_df, test_df, target = get_data_with_transforms()

# Load exp13 Top30
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
base_features = [f for f in exp13_results['best_features'] if f in train_df.columns]

# New non-linear features
new_features = {
    'Age': ['Age_old_penalty'],
    'Height': ['Height_dist_mean', 'Height_dist_mean_sq', 'Height_tall', 'Height_short'],
    'Agility': ['Agility_fast', 'Agility_slow', 'Agility_penalty', 'Agility_Sum_fast', 'Agility_Sum_penalty'],
    'BMI': ['BMI_dist_optimal', 'BMI_dist_optimal_sq', 'BMI_high', 'BMI_low'],
    'JumpEff': ['JumpEff_high', 'JumpEff_bonus'],
    'Combined': ['Young_Fast', 'Old_Slow'],
}

print(f"\nBase features: {len(base_features)}")
print("New non-linear features:")
for category, feats in new_features.items():
    print(f"  {category}: {feats}")

# ==========================================
# CV Tests
# ==========================================
print("\n" + "=" * 60)
print("Cross-Validation Tests")
print("=" * 60)

cv_results = {}

# Baseline
cv_baseline = quick_cv(train_df, target, base_features)
cv_results['baseline'] = cv_baseline
print(f"\n{'Baseline (exp13 Top30)':50} CV: {cv_baseline:.5f}")

# Test each category
for category, feats in new_features.items():
    valid_feats = [f for f in feats if f in train_df.columns]
    test_features = base_features + valid_feats
    cv_score = quick_cv(train_df, target, test_features)
    cv_results[f'baseline_plus_{category}'] = cv_score
    diff = cv_score - cv_baseline
    print(f"{'Baseline + ' + category:50} CV: {cv_score:.5f} ({diff:+.5f})")

# Test all new features
all_new = []
for feats in new_features.values():
    all_new.extend([f for f in feats if f in train_df.columns])
all_new = list(set(all_new))

test_features = base_features + all_new
cv_all = quick_cv(train_df, target, test_features)
cv_results['baseline_plus_all_new'] = cv_all
diff = cv_all - cv_baseline
print(f"{'Baseline + All new features':50} CV: {cv_all:.5f} ({diff:+.5f})")

# Test best combination
# From previous experiments, Age_old_penalty was best
# Try combining best from each category
best_combo = ['Age_old_penalty', 'Agility_penalty', 'Young_Fast']
best_combo = [f for f in best_combo if f in train_df.columns]
test_features = base_features + best_combo
cv_best = quick_cv(train_df, target, test_features)
cv_results['baseline_plus_best_combo'] = cv_best
diff = cv_best - cv_baseline
print(f"{'Baseline + Best combo (Age+Agility+Combined)':50} CV: {cv_best:.5f} ({diff:+.5f})")

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

best_config = max(cv_results.items(), key=lambda x: x[1])
print(f"\nBest configuration: {best_config[0]}")
print(f"Best CV AUC: {best_config[1]:.5f}")
print(f"Improvement over baseline: {best_config[1] - cv_baseline:+.5f}")

# List all new features for potential use
print("\n--- All New Non-linear Features ---")
for f in sorted(all_new):
    print(f"  {f}")

# Save results
results = {
    'new_features': new_features,
    'cv_results': cv_results,
    'best_config': best_config[0],
    'best_cv': best_config[1],
    'all_new_features': all_new
}

with open(f'{EXP_DIR}/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {EXP_DIR}/analysis_results.json")
