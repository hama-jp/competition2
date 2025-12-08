"""
Exp24: Reduce redundant features by correlation analysis
- Find highly correlated pairs within Top31
- Remove lower-importance feature from each pair
- Test if reducing redundancy improves generalization
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

EXP_DIR = '/home/user/competition2/experiments/exp24_reduce_redundancy'
BASE_DIR = '/home/user/competition2'

os.makedirs(EXP_DIR, exist_ok=True)

print("=" * 60)
print("Exp24: Reduce Redundant Features")
print("=" * 60)

# Load exp13 results
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']

# 31 features (best from exp21)
features_31 = top30_features + ['Agility_3cone_Pos_Diff']
print(f"Current features: {len(features_31)}")

# Load exp07 params
with open('/home/user/competition2/experiments/exp07_final/results.json', 'r') as f:
    exp07_results = json.load(f)
lgb_params = exp07_results['best_params_lgb']

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
# Feature Engineering (same as before)
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
# Quick CV (LGB only)
# ==========================================
def quick_cv(train_df, target, features, n_seeds=3):
    seeds = [42, 2023, 101]
    X = train_df[features]
    y = target

    oof = np.zeros(len(train_df))

    for seed in seeds[:n_seeds]:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

            lgb_p = lgb_params.copy()
            lgb_p['random_state'] = seed
            lgb_p['n_estimators'] = 10000
            model = lgb.LGBMClassifier(**lgb_p)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va_idx] += model.predict_proba(X_va)[:, 1] / n_seeds

    return roc_auc_score(y, oof)

# ==========================================
# Get feature importance
# ==========================================
def get_importance(train_df, target, features):
    X = train_df[features]
    y = target

    importance_dict = {f: 0 for f in features}
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(X, y):
        lgb_p = lgb_params.copy()
        lgb_p['n_estimators'] = 500
        model = lgb.LGBMClassifier(**lgb_p)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        for f, imp in zip(features, model.feature_importances_):
            importance_dict[f] += imp / 5

    return importance_dict

# ==========================================
# Load data
# ==========================================
print("\nLoading data...")
train_df, test_df, target, cat_cols = get_data()

# ==========================================
# Correlation analysis within Top31
# ==========================================
print("\n" + "=" * 60)
print("Correlation Analysis within Top31")
print("=" * 60)

# Get numeric features only (exclude categorical)
numeric_features = [f for f in features_31 if f not in cat_cols]
print(f"Numeric features in Top31: {len(numeric_features)}")

# Compute correlation matrix
corr_matrix = train_df[numeric_features].corr()

# Find highly correlated pairs (|corr| > 0.7)
high_corr_pairs = []
for i, f1 in enumerate(numeric_features):
    for j, f2 in enumerate(numeric_features):
        if i < j:
            corr = abs(corr_matrix.loc[f1, f2])
            if corr > 0.7:
                high_corr_pairs.append((f1, f2, corr))

# Sort by correlation
high_corr_pairs.sort(key=lambda x: -x[2])

print(f"\nHighly correlated pairs (|r| > 0.7): {len(high_corr_pairs)}")
print(f"{'Feature 1':35} {'Feature 2':35} {'Corr':>6}")
print("-" * 80)
for f1, f2, corr in high_corr_pairs[:15]:
    print(f"{f1:35} {f2:35} {corr:6.3f}")

# ==========================================
# Get feature importance
# ==========================================
print("\n--- Computing Feature Importance ---")
importance = get_importance(train_df, target, features_31)

# ==========================================
# Identify redundant features to remove
# ==========================================
print("\n" + "=" * 60)
print("Redundant Features to Remove")
print("=" * 60)

# For each highly correlated pair, mark lower-importance one for removal
to_remove = set()
removal_reasons = []

for f1, f2, corr in high_corr_pairs:
    imp1 = importance.get(f1, 0)
    imp2 = importance.get(f2, 0)

    # Remove lower importance one
    if imp1 < imp2:
        remove = f1
        keep = f2
    else:
        remove = f2
        keep = f1

    if remove not in to_remove:
        to_remove.add(remove)
        removal_reasons.append({
            'remove': remove,
            'keep': keep,
            'correlation': corr,
            'remove_imp': importance.get(remove, 0),
            'keep_imp': importance.get(keep, 0)
        })

print(f"\nFeatures to remove: {len(to_remove)}")
print(f"{'Remove':35} {'Keep':35} {'Corr':>6} {'Imp(R)':>8} {'Imp(K)':>8}")
print("-" * 100)
for r in removal_reasons:
    print(f"{r['remove']:35} {r['keep']:35} {r['correlation']:6.3f} {r['remove_imp']:8.1f} {r['keep_imp']:8.1f}")

# ==========================================
# Test removing redundant features
# ==========================================
print("\n" + "=" * 60)
print("Testing Feature Removal")
print("=" * 60)

# Baseline
print("\n--- Baseline (31 features) ---")
cv_baseline = quick_cv(train_df, target, features_31)
print(f"CV: {cv_baseline:.5f}")

# Remove all redundant
features_reduced = [f for f in features_31 if f not in to_remove]
print(f"\n--- Reduced ({len(features_reduced)} features, removed {len(to_remove)}) ---")
cv_reduced = quick_cv(train_df, target, features_reduced)
diff = cv_reduced - cv_baseline
print(f"CV: {cv_reduced:.5f} ({'+' if diff >= 0 else ''}{diff:.5f})")

# Test removing one at a time
print("\n--- Remove one at a time ---")
results_single = {}
for feat in list(to_remove)[:5]:  # Top 5 candidates
    features_test = [f for f in features_31 if f != feat]
    cv = quick_cv(train_df, target, features_test)
    diff = cv - cv_baseline
    print(f"  - {feat:35}: CV = {cv:.5f} ({'+' if diff >= 0 else ''}{diff:.5f})")
    results_single[feat] = {'cv': cv, 'diff': diff}

# Find best single removal
if results_single:
    best_removal = max(results_single.keys(), key=lambda x: results_single[x]['cv'])
    print(f"\n最良の削除: {best_removal} (CV = {results_single[best_removal]['cv']:.5f})")

# Save results
results = {
    'high_corr_pairs': [(f1, f2, float(c)) for f1, f2, c in high_corr_pairs],
    'features_to_remove': list(to_remove),
    'removal_reasons': removal_reasons,
    'cv_baseline': cv_baseline,
    'cv_reduced_all': cv_reduced,
    'single_removal_results': results_single
}

with open(f'{EXP_DIR}/analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nAnalysis saved to {EXP_DIR}/analysis.json")
