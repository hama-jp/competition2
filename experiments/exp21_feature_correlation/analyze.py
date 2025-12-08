"""
Exp21: Feature Correlation Analysis
- Use same feature engineering as exp13
- Analyze correlations between features
- Find features with low correlation to Top30 for potential swaps
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

EXP_DIR = '/home/user/competition2/experiments/exp21_feature_correlation'
BASE_DIR = '/home/user/competition2'

print("=" * 60)
print("Exp21: Feature Correlation Analysis")
print("=" * 60)

# Load exp13 results
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']
print(f"Current Top 30 features: {len(top30_features)}")

# ==========================================
# Target Encoding (from exp07/exp13)
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
# Full Feature Engineering (from exp07/exp13)
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

    exclude_cols = ['Id', 'Drafted', 'is_train']
    features = [c for c in train_df.columns if c not in exclude_cols]
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    return train_df, test_df, target, features, cat_cols

# ==========================================
# Load data
# ==========================================
print("\nLoading and preprocessing data...")
train_df, test_df, target, all_features, cat_cols = get_data()
print(f"Total features: {len(all_features)}")

# Features not in Top30
valid_top30 = [f for f in top30_features if f in all_features]
other_features = [f for f in all_features if f not in valid_top30]
print(f"Valid Top30 features: {len(valid_top30)}")
print(f"Features outside Top30: {len(other_features)}")

# ==========================================
# Feature Importance
# ==========================================
print("\n--- Computing Feature Importance ---")

# Load exp07 best params
with open('/home/user/competition2/experiments/exp07_final/results.json', 'r') as f:
    exp07_results = json.load(f)
lgb_params = exp07_results['best_params_lgb']

X = train_df[all_features].values
y = target.values

importance_dict = {f: 0 for f in all_features}
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    lgb_p = lgb_params.copy()
    lgb_p['n_estimators'] = 500
    model = lgb.LGBMClassifier(**lgb_p)
    model.fit(X_tr, y_tr)

    for f, imp in zip(all_features, model.feature_importances_):
        importance_dict[f] += imp / 5

sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# ==========================================
# Correlation Analysis
# ==========================================
print("\n" + "=" * 60)
print("Correlation Analysis")
print("=" * 60)

# Compute correlation matrix for numeric features
numeric_features = [f for f in all_features if train_df[f].dtype in ['float64', 'int64'] and f not in cat_cols]
corr_matrix = train_df[numeric_features].corr()

# Analyze features outside Top30
print("\n--- High-potential features NOT in Top30 ---")
other_feature_info = []

for feat in other_features:
    if feat in numeric_features:
        # Avg absolute correlation with Top30
        top30_numeric = [f for f in valid_top30 if f in numeric_features]
        if len(top30_numeric) > 0:
            avg_corr = corr_matrix.loc[feat, top30_numeric].abs().mean()
            max_corr = corr_matrix.loc[feat, top30_numeric].abs().max()
        else:
            avg_corr = 0
            max_corr = 0
        imp = importance_dict.get(feat, 0)
        other_feature_info.append({
            'feature': feat,
            'importance': imp,
            'avg_corr_top30': float(avg_corr),
            'max_corr_top30': float(max_corr),
            'score': imp / (avg_corr + 0.1)
        })

# Sort by importance
other_feature_info.sort(key=lambda x: x['importance'], reverse=True)

print("\nTop candidates by importance (outside Top30):")
print(f"{'Rank':>4} {'Feature':35} {'Importance':>10} {'AvgCorr':>8} {'MaxCorr':>8}")
print("-" * 75)
for i, info in enumerate(other_feature_info[:20], 1):
    print(f"{i:4}. {info['feature']:35} {info['importance']:10.1f} {info['avg_corr_top30']:8.3f} {info['max_corr_top30']:8.3f}")

# ==========================================
# Bottom of Top30 (replacement candidates)
# ==========================================
print("\n" + "=" * 60)
print("Bottom 10 of Top30 (candidates for replacement)")
print("=" * 60)

top30_importance = [(f, importance_dict.get(f, 0)) for f in valid_top30]
top30_importance.sort(key=lambda x: x[1])

print(f"{'Feature':35} {'Importance':>10}")
print("-" * 50)
for f, imp in top30_importance[:10]:
    print(f"{f:35} {imp:10.1f}")

# ==========================================
# Find best swap candidates
# ==========================================
print("\n" + "=" * 60)
print("Recommended Single Swaps")
print("=" * 60)

# For each candidate outside Top30, find which Top30 feature it could replace
swap_candidates = []
for add_info in other_feature_info[:10]:
    add_feat = add_info['feature']
    add_imp = add_info['importance']

    # Find the lowest-importance Top30 feature that has high correlation with add_feat
    for remove_feat, remove_imp in top30_importance[:10]:
        if remove_feat in numeric_features and add_feat in numeric_features:
            corr = abs(corr_matrix.loc[add_feat, remove_feat]) if remove_feat in corr_matrix.columns else 0
            # Good swap: adding higher importance and either correlated (replacement) or uncorrelated (diversity)
            imp_gain = add_imp - remove_imp
            swap_candidates.append({
                'add': add_feat,
                'add_imp': add_imp,
                'remove': remove_feat,
                'remove_imp': remove_imp,
                'correlation': corr,
                'imp_gain': imp_gain
            })

# Sort by importance gain
swap_candidates.sort(key=lambda x: x['imp_gain'], reverse=True)

print("\nBest swaps by importance gain:")
print(f"{'Add':35} {'Remove':35} {'ImpGain':>8} {'Corr':>6}")
print("-" * 90)
for swap in swap_candidates[:10]:
    print(f"{swap['add']:35} {swap['remove']:35} {swap['imp_gain']:8.1f} {swap['correlation']:6.3f}")

# ==========================================
# Save results
# ==========================================
results = {
    'all_feature_importance': [(f, float(imp)) for f, imp in sorted_importance],
    'top30_features': valid_top30,
    'bottom10_top30': [(f, float(imp)) for f, imp in top30_importance[:10]],
    'top20_outside_top30': other_feature_info[:20],
    'recommended_swaps': swap_candidates[:10]
}

with open(f'{EXP_DIR}/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nAnalysis saved to {EXP_DIR}/analysis_results.json")
