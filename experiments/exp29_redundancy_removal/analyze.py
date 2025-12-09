"""
Exp29: Chatterjee-based Redundancy Removal
- Identify redundant features in Top30 using Chatterjee correlation
- Remove redundant features and replace with high-value alternatives
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

EXP_DIR = '/home/user/competition2/experiments/exp29_redundancy_removal'
BASE_DIR = '/home/user/competition2'

print("=" * 60)
print("Exp29: Chatterjee-based Redundancy Removal")
print("=" * 60)

# ==========================================
# Chatterjee Correlation
# ==========================================
def chatterjee_correlation(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return np.nan
    order = np.argsort(x)
    y_sorted = y[order]
    r = stats.rankdata(y_sorted, method='average')
    l = np.abs(np.diff(r))
    xi = 1 - (3 * np.sum(l)) / (n**2 - 1)
    return xi


# ==========================================
# Load exp13 Top30 features
# ==========================================
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']
print(f"\nExp13 Top30 features: {len(top30_features)}")

# Load exp27 correlation results
with open('/home/user/competition2/experiments/exp27_chatterjee_correlation/analysis_results.json', 'r') as f:
    exp27_results = json.load(f)

correlation_df = pd.read_csv('/home/user/competition2/experiments/exp27_chatterjee_correlation/correlation_comparison.csv')

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
# Data Preparation
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

    # Age non-linear (from exp28)
    data['Age_old_penalty'] = np.maximum(0, data['Age'] - 23)

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
# Load data
# ==========================================
print("\nLoading data...")
train_df, test_df, target = get_data()

# ==========================================
# Analyze redundancy in Top30
# ==========================================
print("\n" + "=" * 60)
print("Analyzing Redundancy in Top30 Features")
print("=" * 60)

# Compute pairwise Chatterjee for Top30
valid_top30 = [f for f in top30_features if f in train_df.columns]
print(f"\nValid Top30 features: {len(valid_top30)}")

print("\nComputing pairwise Chatterjee correlations...")
redundancy_pairs = []
for i, f1 in enumerate(valid_top30):
    for j, f2 in enumerate(valid_top30):
        if i < j:
            x1 = train_df[f1].values
            x2 = train_df[f2].values
            xi_12 = chatterjee_correlation(x1, x2)
            xi_21 = chatterjee_correlation(x2, x1)
            max_xi = max(xi_12, xi_21)
            if max_xi > 0.7:  # High redundancy threshold
                redundancy_pairs.append({
                    'f1': f1, 'f2': f2,
                    'xi_12': xi_12, 'xi_21': xi_21, 'max_xi': max_xi
                })

redundancy_pairs.sort(key=lambda x: x['max_xi'], reverse=True)

print(f"\n--- High Redundancy Pairs in Top30 (xi > 0.7) ---")
print(f"{'Feature 1':35} {'Feature 2':35} {'Max xi':>10}")
print("-" * 85)
for pair in redundancy_pairs:
    print(f"{pair['f1']:35} {pair['f2']:35} {pair['max_xi']:10.4f}")

# ==========================================
# Identify features to remove
# ==========================================
print("\n" + "=" * 60)
print("Identifying Features to Remove")
print("=" * 60)

# Get feature importance from correlation with target
target_corr = {}
for f in valid_top30:
    xi = chatterjee_correlation(train_df[f].values, target.values)
    target_corr[f] = xi

# For each redundant pair, remove the one with lower target correlation
to_remove = set()
for pair in redundancy_pairs:
    f1, f2 = pair['f1'], pair['f2']
    if f1 in to_remove or f2 in to_remove:
        continue  # Already marked for removal
    # Keep the one with higher correlation to target
    if target_corr.get(f1, 0) >= target_corr.get(f2, 0):
        to_remove.add(f2)
        print(f"Remove: {f2:35} (xi_target={target_corr.get(f2, 0):.4f}) - redundant with {f1}")
    else:
        to_remove.add(f1)
        print(f"Remove: {f1:35} (xi_target={target_corr.get(f1, 0):.4f}) - redundant with {f2}")

print(f"\nFeatures to remove: {len(to_remove)}")

# ==========================================
# Identify replacement candidates
# ==========================================
print("\n" + "=" * 60)
print("Identifying Replacement Candidates")
print("=" * 60)

# Features not in Top30 with high target correlation
all_features = [c for c in train_df.columns if c not in ['Id', 'Drafted', 'is_train']]
not_in_top30 = [f for f in all_features if f not in valid_top30]

# Calculate target correlation for candidates
candidate_scores = []
for f in not_in_top30:
    if train_df[f].dtype in ['float64', 'int64', 'float32', 'int32']:
        xi = chatterjee_correlation(train_df[f].values, target.values)
        # Also check redundancy with remaining Top30
        remaining_top30 = [ff for ff in valid_top30 if ff not in to_remove]
        max_redundancy = 0
        for ff in remaining_top30:
            r = chatterjee_correlation(train_df[f].values, train_df[ff].values)
            if not np.isnan(r):
                max_redundancy = max(max_redundancy, r)
        candidate_scores.append({
            'feature': f,
            'xi_target': xi,
            'max_redundancy': max_redundancy,
            'score': xi - 0.5 * max_redundancy  # Penalize redundancy
        })

candidate_df = pd.DataFrame(candidate_scores)
candidate_df = candidate_df.sort_values('score', ascending=False)

print(f"\n--- Top Replacement Candidates ---")
print(f"{'Feature':40} {'xi_target':>10} {'max_redund':>12} {'Score':>10}")
print("-" * 75)
for _, row in candidate_df.head(15).iterrows():
    print(f"{row['feature']:40} {row['xi_target']:10.4f} {row['max_redundancy']:12.4f} {row['score']:10.4f}")

# ==========================================
# Build optimized feature sets
# ==========================================
print("\n" + "=" * 60)
print("Building Optimized Feature Sets")
print("=" * 60)

# Reduced Top30 (remove redundant)
reduced_features = [f for f in valid_top30 if f not in to_remove]
print(f"\nReduced features: {len(reduced_features)}")

# Add replacement candidates
n_to_add = len(to_remove)
replacements = candidate_df.head(n_to_add * 2)['feature'].tolist()  # Get more candidates
print(f"Replacement candidates: {replacements[:n_to_add]}")

# Also add Age_old_penalty from exp28
if 'Age_old_penalty' not in reduced_features:
    print("Adding Age_old_penalty from exp28")

# ==========================================
# CV Comparison
# ==========================================
print("\n" + "=" * 60)
print("Cross-Validation Comparison")
print("=" * 60)

cv_results = {}

# 1. Baseline (original Top30)
cv_baseline = quick_cv(train_df, target, valid_top30)
cv_results['baseline_top30'] = cv_baseline
print(f"\n{'Baseline Top30':50} CV AUC: {cv_baseline:.5f}")

# 2. Reduced (remove redundant)
cv_reduced = quick_cv(train_df, target, reduced_features)
cv_results['reduced_no_redundant'] = cv_reduced
diff = cv_reduced - cv_baseline
print(f"{'Reduced (remove redundant)':50} CV AUC: {cv_reduced:.5f} ({diff:+.5f})")

# 3. Reduced + Age_old_penalty
features_with_age = reduced_features + ['Age_old_penalty']
cv_age = quick_cv(train_df, target, features_with_age)
cv_results['reduced_plus_age_penalty'] = cv_age
diff = cv_age - cv_baseline
print(f"{'Reduced + Age_old_penalty':50} CV AUC: {cv_age:.5f} ({diff:+.5f})")

# 4. Reduced + top replacements (same count as original)
features_replaced = reduced_features + replacements[:n_to_add]
cv_replaced = quick_cv(train_df, target, features_replaced)
cv_results['reduced_plus_replacements'] = cv_replaced
diff = cv_replaced - cv_baseline
print(f"{'Reduced + top replacements':50} CV AUC: {cv_replaced:.5f} ({diff:+.5f})")

# 5. Reduced + Age_old_penalty + top replacements
features_full = reduced_features + ['Age_old_penalty'] + replacements[:n_to_add]
cv_full = quick_cv(train_df, target, features_full)
cv_results['reduced_plus_age_plus_replacements'] = cv_full
diff = cv_full - cv_baseline
print(f"{'Reduced + Age_penalty + replacements':50} CV AUC: {cv_full:.5f} ({diff:+.5f})")

# 6. Try adding high-Chatterjee features from exp27
high_chatterjee_adds = ['Sprint_40yd_Pos_Z', 'Speed_x_Agility', 'Agility_Sum', 'Talent_Diff']
high_chatterjee_adds = [f for f in high_chatterjee_adds if f not in reduced_features and f in train_df.columns]
features_chatterjee = reduced_features + ['Age_old_penalty'] + high_chatterjee_adds
cv_chatterjee = quick_cv(train_df, target, features_chatterjee)
cv_results['reduced_plus_high_chatterjee'] = cv_chatterjee
diff = cv_chatterjee - cv_baseline
print(f"{'Reduced + Age_penalty + high Chatterjee':50} CV AUC: {cv_chatterjee:.5f} ({diff:+.5f})")

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

print(f"\nRedundant features removed: {sorted(to_remove)}")
print(f"Features in best config: {len(features_chatterjee)}")

# Save results
results = {
    'redundancy_pairs': redundancy_pairs,
    'removed_features': list(to_remove),
    'replacement_candidates': candidate_df.head(20).to_dict(orient='records'),
    'cv_results': cv_results,
    'best_config': best_config[0],
    'best_cv': best_config[1],
    'reduced_features': reduced_features,
    'final_features': features_chatterjee
}

with open(f'{EXP_DIR}/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {EXP_DIR}/analysis_results.json")
