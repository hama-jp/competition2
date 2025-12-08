"""
Exp18: Stacking Ensemble
- Layer 1: LGB, XGB, CAT trained separately
- Layer 2: Meta-model using Layer 1 OOF predictions
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

BASE_DIR = '/home/user/competition2'
EXP_DIR = '/home/user/competition2/experiments/exp18_stacking'

print("=" * 60)
print("Exp18: Stacking Ensemble")
print("=" * 60)

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
# Load params and features
# ==========================================
with open('/home/user/competition2/experiments/exp07_final/results.json', 'r') as f:
    exp07_results = json.load(f)

lgb_params = exp07_results['best_params_lgb']
xgb_params = exp07_results['best_params_xgb']
cat_params = exp07_results['best_params_cat']

with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)

top_30_features = exp13_results['best_features']

# ==========================================
# Main
# ==========================================
print("Loading data...")
train_df, test_df, target = get_data()

features = [f for f in top_30_features if f in train_df.columns]
cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']
cat_indices = [features.index(c) for c in cat_cols if c in features]

print(f"Features: {len(features)}")

X_train = train_df[features]
y_train = target
X_test = test_df[features]

# ==========================================
# Layer 1: Train base models and get OOF predictions
# ==========================================
print("\n" + "=" * 60)
print("Layer 1: Training base models")
print("=" * 60)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))

pred_lgb = np.zeros(len(test_df))
pred_xgb = np.zeros(len(test_df))
pred_cat = np.zeros(len(test_df))

for seed_idx, seed in enumerate(SEEDS):
    print(f"  Seed {seed_idx+1}/{N_SEEDS}: {seed}")
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

cv_lgb = roc_auc_score(y_train, oof_lgb)
cv_xgb = roc_auc_score(y_train, oof_xgb)
cv_cat = roc_auc_score(y_train, oof_cat)

print(f"\nLayer 1 CVs:")
print(f"  LGB: {cv_lgb:.5f}")
print(f"  XGB: {cv_xgb:.5f}")
print(f"  CAT: {cv_cat:.5f}")

# Simple average baseline
oof_avg = (oof_lgb + oof_xgb + oof_cat) / 3
pred_avg = (pred_lgb + pred_xgb + pred_cat) / 3
cv_avg = roc_auc_score(y_train, oof_avg)
print(f"  Simple Avg: {cv_avg:.5f}")

# ==========================================
# Layer 2: Meta-model using Layer 1 predictions
# ==========================================
print("\n" + "=" * 60)
print("Layer 2: Training meta-models")
print("=" * 60)

# Create stacking features
X_stack_train = pd.DataFrame({
    'lgb': oof_lgb,
    'xgb': oof_xgb,
    'cat': oof_cat
})
X_stack_test = pd.DataFrame({
    'lgb': pred_lgb,
    'xgb': pred_xgb,
    'cat': pred_cat
})

# Add original features for hybrid stacking
X_stack_hybrid_train = pd.concat([X_stack_train, X_train.reset_index(drop=True)], axis=1)
X_stack_hybrid_test = pd.concat([X_stack_test, X_test.reset_index(drop=True)], axis=1)

results = {}

# Method 1: Logistic Regression on predictions only
print("\n--- Method 1: LogisticRegression (predictions only) ---")
oof_lr = np.zeros(len(train_df))
pred_lr = np.zeros(len(test_df))

for seed in SEEDS:
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_stack_train, y_train)):
        X_tr = X_stack_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_stack_train.iloc[va_idx]

        model = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)
        model.fit(X_tr, y_tr)

        oof_lr[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_lr += model.predict_proba(X_stack_test)[:, 1] / (N_FOLDS * N_SEEDS)

cv_lr = roc_auc_score(y_train, oof_lr)
print(f"CV: {cv_lr:.5f}")
results['LR_pred_only'] = {'cv': cv_lr, 'pred': pred_lr}

# Method 2: LightGBM on predictions only
print("\n--- Method 2: LightGBM (predictions only) ---")
oof_lgb_meta = np.zeros(len(train_df))
pred_lgb_meta = np.zeros(len(test_df))

for seed in SEEDS:
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_stack_train, y_train)):
        X_tr = X_stack_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_stack_train.iloc[va_idx]

        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=8,
            max_depth=3,
            random_state=seed,
            verbosity=-1
        )
        model.fit(X_tr, y_tr)

        oof_lgb_meta[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_lgb_meta += model.predict_proba(X_stack_test)[:, 1] / (N_FOLDS * N_SEEDS)

cv_lgb_meta = roc_auc_score(y_train, oof_lgb_meta)
print(f"CV: {cv_lgb_meta:.5f}")
results['LGB_pred_only'] = {'cv': cv_lgb_meta, 'pred': pred_lgb_meta}

# Method 3: LightGBM on predictions + original features (hybrid)
print("\n--- Method 3: LightGBM (predictions + features) ---")
oof_hybrid = np.zeros(len(train_df))
pred_hybrid = np.zeros(len(test_df))

for seed in SEEDS:
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_stack_hybrid_train, y_train)):
        X_tr = X_stack_hybrid_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_stack_hybrid_train.iloc[va_idx]

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=16,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed,
            verbosity=-1
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_train.iloc[va_idx])],
                 callbacks=[lgb.early_stopping(50, verbose=False)])

        oof_hybrid[va_idx] += model.predict_proba(X_va)[:, 1] / N_SEEDS
        pred_hybrid += model.predict_proba(X_stack_hybrid_test)[:, 1] / (N_FOLDS * N_SEEDS)

cv_hybrid = roc_auc_score(y_train, oof_hybrid)
print(f"CV: {cv_hybrid:.5f}")
results['LGB_hybrid'] = {'cv': cv_hybrid, 'pred': pred_hybrid}

# Method 4: Weighted blend of stacking + simple average
print("\n--- Method 4: Blend (Stacking + Simple Avg) ---")
best_blend_cv = 0
best_blend_weight = 0
for w in np.arange(0, 1.05, 0.1):
    oof_blend = w * oof_hybrid + (1-w) * oof_avg
    cv_blend = roc_auc_score(y_train, oof_blend)
    if cv_blend > best_blend_cv:
        best_blend_cv = cv_blend
        best_blend_weight = w

pred_blend = best_blend_weight * pred_hybrid + (1-best_blend_weight) * pred_avg
print(f"Best blend: {best_blend_weight:.1f} * Hybrid + {1-best_blend_weight:.1f} * Avg")
print(f"CV: {best_blend_cv:.5f}")
results['Blend'] = {'cv': best_blend_cv, 'pred': pred_blend}

# ==========================================
# Results Summary
# ==========================================
print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)

print(f"\nBaseline (Simple Avg): {cv_avg:.5f}")
for name, res in results.items():
    diff = res['cv'] - cv_avg
    sign = "+" if diff >= 0 else ""
    print(f"  {name:20}: {res['cv']:.5f} ({sign}{diff:.5f})")

# Find best
best_name = max(results, key=lambda x: results[x]['cv'])
best_cv = results[best_name]['cv']
best_pred = results[best_name]['pred']

print(f"\nBest: {best_name} with CV = {best_cv:.5f}")
print(f"Exp13 reference: CV = 0.85138, LB = 0.84524")

# Save submissions
for name, res in results.items():
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'Drafted': res['pred']
    })
    submission.to_csv(f'{EXP_DIR}/submission_{name}.csv', index=False)

# Save simple average as baseline
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': pred_avg
})
submission.to_csv(f'{EXP_DIR}/submission_simple_avg.csv', index=False)

# Save best
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': best_pred
})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

# Save results
results_summary = {
    'cv_simple_avg': cv_avg,
    'stacking_results': {k: v['cv'] for k, v in results.items()},
    'best_method': best_name,
    'best_cv': best_cv,
    'blend_weight': best_blend_weight,
    'exp13_cv': 0.85138,
    'exp13_lb': 0.84524
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to {EXP_DIR}")
