"""
Exp13: Feature Selection - Reduce features to improve generalization
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5  # Use 5 seeds for faster iteration
SEEDS = [42, 2023, 101, 555, 999]

BASE_DIR = '/home/user/competition2'
EXP_DIR = '/home/user/competition2/experiments/exp13_feature_selection'

print("=" * 60)
print("Exp13: Feature Selection")
print("=" * 60)

# ==========================================
# Target Encoding (from exp07)
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
# Data Preparation (from exp07)
# ==========================================
def get_data():
    print("Loading and preprocessing...")
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
# Load exp07 best params
# ==========================================
with open('/home/user/competition2/experiments/exp07_final/results.json', 'r') as f:
    exp07_results = json.load(f)

lgb_params = exp07_results['best_params_lgb']
xgb_params = exp07_results['best_params_xgb']
cat_params = exp07_results['best_params_cat']

# ==========================================
# Get Feature Importance
# ==========================================
def get_feature_importance(train_df, target, features):
    """Get feature importance from LightGBM"""
    print("\nCalculating feature importance...")

    X = train_df[features]
    y = target

    importance_dict = {f: 0 for f in features}

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]

        lgb_p = lgb_params.copy()
        lgb_p['n_estimators'] = 500
        model = lgb.LGBMClassifier(**lgb_p)
        model.fit(X_tr, y_tr)

        for f, imp in zip(features, model.feature_importances_):
            importance_dict[f] += imp / 5

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_features

# ==========================================
# Train with selected features
# ==========================================
def train_model(train_df, test_df, target, features, cat_cols):
    """Train ensemble with selected features"""

    cat_indices = [features.index(c) for c in cat_cols if c in features]

    X_train = train_df[features]
    y_train = target
    X_test = test_df[features]

    oof_lgb = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    pred_lgb = np.zeros(len(test_df))
    pred_xgb = np.zeros(len(test_df))
    pred_cat = np.zeros(len(test_df))

    for seed in SEEDS:
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
            oof_lgb[va_idx] += model_lgb.predict_proba(X_va)[:, 1] / len(SEEDS)
            pred_lgb += model_lgb.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

            # XGB
            xgb_p = xgb_params.copy()
            xgb_p['random_state'] = seed
            model_xgb = xgb.XGBClassifier(**xgb_p, n_estimators=10000)
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_xgb[va_idx] += model_xgb.predict_proba(X_va)[:, 1] / len(SEEDS)
            pred_xgb += model_xgb.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

            # CatBoost
            cat_p = cat_params.copy()
            cat_p['random_seed'] = seed
            model_cat = CatBoostClassifier(**cat_p)
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)
            model_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
            oof_cat[va_idx] += model_cat.predict_proba(X_va)[:, 1] / len(SEEDS)
            pred_cat += model_cat.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

    # Ensemble
    oof_final = (oof_lgb * 0.3 + oof_xgb * 0.3 + oof_cat * 0.4)
    pred_final = (pred_lgb * 0.3 + pred_xgb * 0.3 + pred_cat * 0.4)
    cv_final = roc_auc_score(y_train, oof_final)

    return oof_final, pred_final, cv_final

# ==========================================
# Main
# ==========================================
# Load data
train_df, test_df, target, all_features, cat_cols = get_data()
print(f"Total features: {len(all_features)}")

# Get feature importance
sorted_features = get_feature_importance(train_df, target, all_features)

# Print top 30 features
print("\nTop 30 features by importance:")
for i, (feat, imp) in enumerate(sorted_features[:30]):
    print(f"  {i+1:2}. {feat:40} {imp:.1f}")

# Save feature importance
with open(f'{EXP_DIR}/feature_importance.json', 'w') as f:
    json.dump(sorted_features, f, indent=2)

# Try different feature counts
results = {}
feature_counts = [20, 30, 40, 50, 60, 70, 80, 92]  # 92 is all features

print("\n" + "=" * 60)
print("Testing different feature counts")
print("=" * 60)

best_cv = 0
best_n = 0
best_pred = None

for n_features in feature_counts:
    print(f"\n--- Top {n_features} features ---")

    # Select top N features
    selected_features = [f for f, _ in sorted_features[:n_features]]

    # Train
    oof, pred, cv = train_model(train_df, test_df, target, selected_features, cat_cols)

    print(f"CV: {cv:.5f}")
    results[n_features] = cv

    if cv > best_cv:
        best_cv = cv
        best_n = n_features
        best_pred = pred
        best_features = selected_features

# ==========================================
# Results
# ==========================================
print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)

for n, cv in sorted(results.items()):
    marker = " <-- BEST" if n == best_n else ""
    print(f"  {n:3} features: CV = {cv:.5f}{marker}")

print(f"\nExp07 (92 features): CV = 0.84651, LB = 0.84267")
print(f"Best: {best_n} features with CV = {best_cv:.5f}")

# Save best submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': best_pred
})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

# Save results
results_summary = {
    'feature_count_results': results,
    'best_n_features': best_n,
    'best_cv': best_cv,
    'best_features': best_features,
    'exp07_cv': 0.84651,
    'exp07_lb': 0.84267
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nSubmission saved to {EXP_DIR}/submission.csv")
