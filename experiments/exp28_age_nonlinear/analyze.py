"""
Exp28: Age Non-linear Transformation Analysis
- Analyze relationship between Age and Draft probability
- Test multiple non-linear transformations
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

EXP_DIR = '/home/user/competition2/experiments/exp28_age_nonlinear'
BASE_DIR = '/home/user/competition2'

print("=" * 60)
print("Exp28: Age Non-linear Transformation Analysis")
print("=" * 60)

# ==========================================
# Load raw data
# ==========================================
train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
print(f"\nTraining samples: {len(train)}")

# ==========================================
# Analyze Age distribution and Draft rate
# ==========================================
print("\n" + "=" * 60)
print("Age Distribution and Draft Rate Analysis")
print("=" * 60)

age_stats = train.groupby('Age').agg({
    'Drafted': ['count', 'sum', 'mean']
}).round(3)
age_stats.columns = ['count', 'drafted', 'draft_rate']
age_stats['pct_of_total'] = (age_stats['count'] / len(train) * 100).round(1)

print("\n--- Draft Rate by Age ---")
print(f"{'Age':>4} {'Count':>8} {'Drafted':>8} {'Rate':>8} {'%Total':>8}")
print("-" * 45)
for age in sorted(train['Age'].dropna().unique()):
    row = age_stats.loc[age]
    print(f"{age:4.0f} {int(row['count']):8} {int(row['drafted']):8} {row['draft_rate']:8.3f} {row['pct_of_total']:7.1f}%")

# Calculate optimal age (highest draft rate with sufficient samples)
min_samples = 50
valid_ages = age_stats[age_stats['count'] >= min_samples]
optimal_age = valid_ages['draft_rate'].idxmax()
print(f"\nOptimal age (highest draft rate with n>={min_samples}): {optimal_age}")
print(f"Draft rate at optimal age: {valid_ages.loc[optimal_age, 'draft_rate']:.3f}")

# Mean age
mean_age = train['Age'].mean()
median_age = train['Age'].median()
print(f"\nMean age: {mean_age:.2f}")
print(f"Median age: {median_age:.2f}")

# ==========================================
# Correlation Analysis
# ==========================================
print("\n" + "=" * 60)
print("Correlation Analysis")
print("=" * 60)

# Chatterjee correlation function
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

# Remove NaN for correlation
valid_mask = ~train['Age'].isna()
age = train.loc[valid_mask, 'Age'].values
target = train.loc[valid_mask, 'Drafted'].values

pearson_r, _ = stats.pearsonr(age, target)
spearman_r, _ = stats.spearmanr(age, target)
chatterjee_xi = chatterjee_correlation(age, target)

print(f"\nAge vs Drafted correlations:")
print(f"  Pearson:    {pearson_r:.4f}")
print(f"  Spearman:   {spearman_r:.4f}")
print(f"  Chatterjee: {chatterjee_xi:.4f}")
print(f"  Chatterjee - Pearson = {chatterjee_xi - abs(pearson_r):.4f} (non-linearity indicator)")

# ==========================================
# Non-linear Transformation Candidates
# ==========================================
print("\n" + "=" * 60)
print("Non-linear Transformation Candidates")
print("=" * 60)

transformations = {}

# 1. Polynomial: Age^2, Age^3
transformations['Age_squared'] = age ** 2
transformations['Age_cubed'] = age ** 3

# 2. Distance from optimal age (quadratic penalty)
transformations['Age_dist_optimal'] = np.abs(age - optimal_age)
transformations['Age_dist_optimal_sq'] = (age - optimal_age) ** 2

# 3. Distance from mean
transformations['Age_dist_mean'] = np.abs(age - mean_age)
transformations['Age_dist_mean_sq'] = (age - mean_age) ** 2

# 4. Binning approaches
age_bins_3 = pd.cut(age, bins=[0, 21, 23, 100], labels=[0, 1, 2])
age_bins_4 = pd.cut(age, bins=[0, 21, 22, 23, 100], labels=[0, 1, 2, 3])
age_bins_5 = pd.cut(age, bins=[0, 20, 21, 22, 23, 100], labels=[0, 1, 2, 3, 4])
transformations['Age_bin_3'] = age_bins_3.astype(float)
transformations['Age_bin_4'] = age_bins_4.astype(float)
transformations['Age_bin_5'] = age_bins_5.astype(float)

# 5. Optimal age indicator
transformations['Age_is_optimal'] = (age == optimal_age).astype(float)
transformations['Age_near_optimal'] = ((age >= optimal_age - 1) & (age <= optimal_age + 1)).astype(float)

# 6. Young/Old indicators
transformations['Age_young'] = (age <= 21).astype(float)
transformations['Age_old'] = (age >= 24).astype(float)
transformations['Age_prime'] = ((age >= 21) & (age <= 23)).astype(float)

# 7. Inverse and log
transformations['Age_inverse'] = 1 / age
transformations['Age_log'] = np.log(age)

# 8. Spline-like: piecewise linear
transformations['Age_young_penalty'] = np.maximum(0, 22 - age)  # Penalty for being young
transformations['Age_old_penalty'] = np.maximum(0, age - 23)    # Penalty for being old

# 9. Gaussian/bell curve centered at optimal
sigma = 1.5
transformations['Age_gaussian'] = np.exp(-((age - optimal_age) ** 2) / (2 * sigma ** 2))

# 10. Interaction with experience proxy
transformations['Age_minus_20'] = np.maximum(0, age - 20)  # Years since typical college entry

print("\n--- Correlation of Each Transformation with Target ---")
print(f"{'Transformation':30} {'Pearson':>10} {'Spearman':>10} {'Chatterjee':>10}")
print("-" * 65)

transform_results = []
for name, values in transformations.items():
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 100:
        continue
    p_r, _ = stats.pearsonr(values[valid_mask], target[valid_mask])
    s_r, _ = stats.spearmanr(values[valid_mask], target[valid_mask])
    c_xi = chatterjee_correlation(values, target)
    transform_results.append({
        'name': name,
        'pearson': abs(p_r),
        'spearman': abs(s_r),
        'chatterjee': c_xi,
        'pearson_signed': p_r
    })
    print(f"{name:30} {p_r:10.4f} {s_r:10.4f} {c_xi:10.4f}")

# Sort by absolute Pearson (linear relationship strength)
transform_df = pd.DataFrame(transform_results)
transform_df = transform_df.sort_values('pearson', ascending=False)

print("\n--- Top Transformations by |Pearson| ---")
for _, row in transform_df.head(10).iterrows():
    print(f"  {row['name']:30} Pearson: {row['pearson_signed']:+.4f}")

# ==========================================
# Cross-Validation Test
# ==========================================
print("\n" + "=" * 60)
print("Cross-Validation: Comparing Age Transformations")
print("=" * 60)

# Load full feature engineering
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


def get_base_data():
    """Get base features (without age transformations)"""
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

    # Original age interactions (will test replacements)
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


def add_age_transforms(train_df, test_df, optimal_age=22):
    """Add various age transformations"""
    for df in [train_df, test_df]:
        age = df['Age'].values

        # Distance-based
        df['Age_dist_optimal'] = np.abs(age - optimal_age)
        df['Age_dist_optimal_sq'] = (age - optimal_age) ** 2

        # Polynomial
        df['Age_squared'] = age ** 2

        # Indicators
        df['Age_prime'] = ((age >= 21) & (age <= 23)).astype(float)
        df['Age_young'] = (age <= 21).astype(float)
        df['Age_old'] = (age >= 24).astype(float)

        # Piecewise
        df['Age_young_penalty'] = np.maximum(0, 22 - age)
        df['Age_old_penalty'] = np.maximum(0, age - 23)

        # Gaussian
        sigma = 1.5
        df['Age_gaussian'] = np.exp(-((age - optimal_age) ** 2) / (2 * sigma ** 2))

    return train_df, test_df


# Load base data
print("\nLoading base data...")
train_df, test_df, target = get_base_data()

# Add age transformations
train_df, test_df = add_age_transforms(train_df, test_df, optimal_age=optimal_age)

# Define feature sets to test
# Base features from exp13 (30 features)
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
base_features = [f for f in exp13_results['best_features'] if f in train_df.columns]

# New age features
new_age_features = [
    'Age_dist_optimal', 'Age_dist_optimal_sq', 'Age_squared',
    'Age_prime', 'Age_young', 'Age_old',
    'Age_young_penalty', 'Age_old_penalty', 'Age_gaussian'
]

print(f"\nBase features: {len(base_features)}")
print(f"New age features: {new_age_features}")

# ==========================================
# Quick CV function (LightGBM only for speed)
# ==========================================
def quick_cv(train_df, target, features, n_folds=5, seed=42):
    """Quick CV with LightGBM only"""
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
# Test different feature configurations
# ==========================================
print("\n--- CV Results ---")
print(f"{'Configuration':50} {'CV AUC':>10}")
print("-" * 65)

cv_results = {}

# 1. Baseline (original features)
cv_baseline = quick_cv(train_df, target, base_features)
cv_results['baseline'] = cv_baseline
print(f"{'Baseline (exp13 Top30)':50} {cv_baseline:.5f}")

# 2. Remove original Age, add each new feature
age_original_features = ['Age', 'Age_x_Speed', 'Age_x_Momentum', 'Age_div_Explosion', 'Age_Year_Diff']
features_no_age = [f for f in base_features if f not in age_original_features]

for new_feat in new_age_features:
    test_features = features_no_age + [new_feat]
    cv_score = quick_cv(train_df, target, test_features)
    cv_results[f'replace_age_with_{new_feat}'] = cv_score
    diff = cv_score - cv_baseline
    print(f"{'No Age + ' + new_feat:50} {cv_score:.5f} ({diff:+.5f})")

# 3. Add new features to baseline (keep original Age features)
for new_feat in new_age_features:
    test_features = base_features + [new_feat]
    cv_score = quick_cv(train_df, target, test_features)
    cv_results[f'baseline_plus_{new_feat}'] = cv_score
    diff = cv_score - cv_baseline
    print(f"{'Baseline + ' + new_feat:50} {cv_score:.5f} ({diff:+.5f})")

# 4. Best combination: baseline + multiple new features
best_new = ['Age_dist_optimal_sq', 'Age_gaussian', 'Age_prime']
test_features = base_features + best_new
cv_score = quick_cv(train_df, target, test_features)
cv_results['baseline_plus_best_combo'] = cv_score
diff = cv_score - cv_baseline
print(f"{'Baseline + dist_sq + gaussian + prime':50} {cv_score:.5f} ({diff:+.5f})")

# 5. All new age features
test_features = base_features + new_age_features
cv_score = quick_cv(train_df, target, test_features)
cv_results['baseline_plus_all_new'] = cv_score
diff = cv_score - cv_baseline
print(f"{'Baseline + all new age features':50} {cv_score:.5f} ({diff:+.5f})")

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Find best configuration
best_config = max(cv_results.items(), key=lambda x: x[1])
print(f"\nBest configuration: {best_config[0]}")
print(f"Best CV AUC: {best_config[1]:.5f}")
print(f"Improvement over baseline: {best_config[1] - cv_baseline:+.5f}")

# Save results
results = {
    'optimal_age': int(optimal_age),
    'age_stats': age_stats.to_dict(),
    'transformation_correlations': transform_df.to_dict(orient='records'),
    'cv_results': cv_results,
    'best_config': best_config[0],
    'best_cv': best_config[1],
    'new_age_features': new_age_features
}

with open(f'{EXP_DIR}/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {EXP_DIR}/analysis_results.json")
