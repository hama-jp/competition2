"""
Exp21: Test specific feature swaps
- Try adding low-correlation features one at a time
- Use LightGBM only for fast testing
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
print("Exp21: Test Feature Swaps")
print("=" * 60)

# Load exp13 results
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']

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

    return train_df, test_df, target

# ==========================================
# Quick CV function (LGB only, 1 seed)
# ==========================================
def quick_cv(train_df, target, features):
    X = train_df[features]
    y = target

    oof = np.zeros(len(train_df))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        lgb_p = lgb_params.copy()
        lgb_p['n_estimators'] = 10000
        model = lgb.LGBMClassifier(**lgb_p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return roc_auc_score(y, oof)

# ==========================================
# Load data
# ==========================================
print("\nLoading data...")
train_df, test_df, target = get_data()

# Baseline with Top30
print("\n--- Baseline (Top30) ---")
cv_baseline = quick_cv(train_df, target, top30_features)
print(f"CV: {cv_baseline:.5f}")

# ==========================================
# Test adding features (31 features)
# ==========================================
print("\n" + "=" * 60)
print("Testing: Add 1 feature (31 total)")
print("=" * 60)

# Features to try adding (low correlation with Top30)
add_candidates = [
    'School_Year_Count',      # avg_corr=0.072
    'Shuttle_Pos_Z',          # avg_corr=0.125
    'Agility_3cone_Pos_Diff', # avg_corr=0.143
    'Shuttle_Type_Z',         # avg_corr=0.148
    'Sprint_40yd_Pos_Diff',   # avg_corr=0.217
]

results_add = {}
for feat in add_candidates:
    features_31 = top30_features + [feat]
    cv = quick_cv(train_df, target, features_31)
    diff = cv - cv_baseline
    sign = "+" if diff >= 0 else ""
    print(f"  + {feat:30}: CV = {cv:.5f} ({sign}{diff:.5f})")
    results_add[feat] = cv

# ==========================================
# Test swapping features
# ==========================================
print("\n" + "=" * 60)
print("Testing: Swap 1 feature")
print("=" * 60)

# Remove candidates (bottom of Top30)
remove_candidates = ['Position_Type_TE', 'Broad_Jump_Year_Rank', 'Sprint_Efficiency']

results_swap = {}
for add_feat in add_candidates[:3]:  # Top 3 to add
    for remove_feat in remove_candidates:
        features_swap = [f for f in top30_features if f != remove_feat] + [add_feat]
        cv = quick_cv(train_df, target, features_swap)
        diff = cv - cv_baseline
        sign = "+" if diff >= 0 else ""
        print(f"  +{add_feat:25} -{remove_feat:20}: CV = {cv:.5f} ({sign}{diff:.5f})")
        results_swap[f"+{add_feat},-{remove_feat}"] = cv

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"\nBaseline (Top30): CV = {cv_baseline:.5f}")

print("\nBest additions:")
for feat, cv in sorted(results_add.items(), key=lambda x: -x[1])[:3]:
    diff = cv - cv_baseline
    print(f"  + {feat}: {cv:.5f} ({'+' if diff >= 0 else ''}{diff:.5f})")

print("\nBest swaps:")
for swap, cv in sorted(results_swap.items(), key=lambda x: -x[1])[:3]:
    diff = cv - cv_baseline
    print(f"  {swap}: {cv:.5f} ({'+' if diff >= 0 else ''}{diff:.5f})")

# Save results
results = {
    'baseline_cv': cv_baseline,
    'add_results': results_add,
    'swap_results': results_swap
}
with open(f'{EXP_DIR}/swap_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {EXP_DIR}/swap_test_results.json")
