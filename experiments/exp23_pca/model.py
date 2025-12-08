"""
Exp23: Add PCA components as features
- Apply PCA on numeric features
- Add top N components to the 31 features
- Test different N values
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

EXP_DIR = '/home/user/competition2/experiments/exp23_pca'
BASE_DIR = '/home/user/competition2'

print("=" * 60)
print("Exp23: Add PCA Components")
print("=" * 60)

# Load exp13 results
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']

# 31 features (best from exp21)
base_features = top30_features + ['Agility_3cone_Pos_Diff']
print(f"Base features: {len(base_features)}")

# Load exp07 params
with open('/home/user/competition2/experiments/exp07_final/results.json', 'r') as f:
    exp07_results = json.load(f)
lgb_params = exp07_results['best_params_lgb']
xgb_params = exp07_results['best_params_xgb']
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
# Load data
# ==========================================
print("\nLoading data...")
train_df, test_df, target, cat_cols = get_data()

# Get all numeric features for PCA (excluding categorical)
all_features = [c for c in train_df.columns if c not in ['Id', 'Drafted', 'is_train'] + cat_cols]
numeric_for_pca = [f for f in all_features if train_df[f].dtype in ['float64', 'int64']]
print(f"Numeric features for PCA: {len(numeric_for_pca)}")

# ==========================================
# Apply PCA
# ==========================================
print("\n--- Applying PCA ---")

# Prepare data for PCA
X_pca_train = train_df[numeric_for_pca].fillna(0)
X_pca_test = test_df[numeric_for_pca].fillna(0)

# Standardize
scaler = StandardScaler()
X_pca_train_scaled = scaler.fit_transform(X_pca_train)
X_pca_test_scaled = scaler.transform(X_pca_test)

# Fit PCA
pca = PCA(n_components=20)
pca_train = pca.fit_transform(X_pca_train_scaled)
pca_test = pca.transform(X_pca_test_scaled)

print(f"Explained variance ratio (top 10): {pca.explained_variance_ratio_[:10].round(3)}")
print(f"Cumulative variance (top 10): {np.cumsum(pca.explained_variance_ratio_[:10]).round(3)}")

# Add PCA components to dataframes
for i in range(20):
    train_df[f'PCA_{i+1}'] = pca_train[:, i]
    test_df[f'PCA_{i+1}'] = pca_test[:, i]

# ==========================================
# Test different number of PCA components
# ==========================================
print("\n" + "=" * 60)
print("Testing different numbers of PCA components")
print("=" * 60)

# Baseline without PCA
print("\n--- Baseline (31 features, no PCA) ---")
cv_baseline = quick_cv(train_df, target, base_features)
print(f"CV: {cv_baseline:.5f}")

# Test adding 1, 2, 3, 5, 10 PCA components
pca_counts = [1, 2, 3, 5, 10]
results = {}

for n_pca in pca_counts:
    pca_features = [f'PCA_{i+1}' for i in range(n_pca)]
    features = base_features + pca_features
    cv = quick_cv(train_df, target, features)
    diff = cv - cv_baseline
    sign = "+" if diff >= 0 else ""
    print(f"  31 + {n_pca} PCA: CV = {cv:.5f} ({sign}{diff:.5f})")
    results[n_pca] = {'cv': cv, 'diff': diff}

# Find best
best_n_pca = max(results.keys(), key=lambda x: results[x]['cv'])
best_cv = results[best_n_pca]['cv']

print(f"\n最良: {best_n_pca} PCA成分 (CV = {best_cv:.5f})")

# ==========================================
# Full ensemble with best PCA count
# ==========================================
if best_cv > cv_baseline:
    print("\n" + "=" * 60)
    print(f"Full ensemble with {best_n_pca} PCA components")
    print("=" * 60)

    pca_features = [f'PCA_{i+1}' for i in range(best_n_pca)]
    features = base_features + pca_features
    print(f"Total features: {len(features)}")

    X_train = train_df[features]
    y_train = target
    X_test = test_df[features]

    # Update cat_indices for new feature list
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    oof_lgb = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    pred_lgb = np.zeros(len(test_df))
    pred_xgb = np.zeros(len(test_df))
    pred_cat = np.zeros(len(test_df))

    for seed_idx, seed in enumerate(SEEDS):
        print(f"Seed {seed} ({seed_idx+1}/{N_SEEDS})...")
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

            # LGB
            lgb_p = lgb_params.copy()
            lgb_p['random_state'] = seed
            model_lgb = lgb.LGBMClassifier(**lgb_p, n_estimators=10000)
            model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_lgb[va_idx] += model_lgb.predict_proba(X_va)[:, 1] / N_SEEDS
            pred_lgb += model_lgb.predict_proba(X_test)[:, 1] / (N_FOLDS * N_SEEDS)

            # XGB
            xgb_p = xgb_params.copy()
            xgb_p['random_state'] = seed
            model_xgb = xgb.XGBClassifier(**xgb_p, n_estimators=10000)
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_xgb[va_idx] += model_xgb.predict_proba(X_va)[:, 1] / N_SEEDS
            pred_xgb += model_xgb.predict_proba(X_test)[:, 1] / (N_FOLDS * N_SEEDS)

            # CatBoost
            cat_p = cat_params.copy()
            cat_p['random_seed'] = seed
            model_cat = CatBoostClassifier(**cat_p)
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)
            model_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
            oof_cat[va_idx] += model_cat.predict_proba(X_va)[:, 1] / N_SEEDS
            pred_cat += model_cat.predict_proba(X_test)[:, 1] / (N_FOLDS * N_SEEDS)

    # Final scores
    cv_lgb = roc_auc_score(y_train, oof_lgb)
    cv_xgb = roc_auc_score(y_train, oof_xgb)
    cv_cat = roc_auc_score(y_train, oof_cat)

    # Ensemble
    oof_final = oof_lgb * 0.3 + oof_xgb * 0.3 + oof_cat * 0.4
    pred_final = pred_lgb * 0.3 + pred_xgb * 0.3 + pred_cat * 0.4
    cv_final = roc_auc_score(y_train, oof_final)

    print(f"\nResults:")
    print(f"  LGB CV: {cv_lgb:.5f}")
    print(f"  XGB CV: {cv_xgb:.5f}")
    print(f"  CAT CV: {cv_cat:.5f}")
    print(f"  Ensemble CV: {cv_final:.5f}")
    print(f"\n  exp21 (31 features): CV = 0.85161, LB = 0.84582")
    print(f"  Difference: {cv_final - 0.85161:+.5f}")

    # Save submission
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'Drafted': pred_final
    })
    submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)
    print(f"\nSubmission saved to {EXP_DIR}/submission.csv")

    # Save results
    results_summary = {
        'best_n_pca': best_n_pca,
        'n_features': len(features),
        'features': features,
        'cv_lgb': cv_lgb,
        'cv_xgb': cv_xgb,
        'cv_cat': cv_cat,
        'cv_ensemble': cv_final,
        'pca_test_results': {str(k): v for k, v in results.items()},
        'exp21_cv': 0.85161,
        'exp21_lb': 0.84582
    }
else:
    print("\nPCA did not improve CV. Skipping full ensemble.")
    results_summary = {
        'best_n_pca': 0,
        'pca_test_results': {str(k): v for k, v in results.items()},
        'baseline_cv': cv_baseline,
        'conclusion': 'PCA did not improve CV'
    }

with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to {EXP_DIR}/results.json")
