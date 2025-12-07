"""
Exp16: Diverse Ensemble - Add ExtraTrees and RandomForest
- Base: 30 features from exp13 (best LB)
- Add: ExtraTrees, RandomForest for diversity
- 5 seeds
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

BASE_DIR = '/home/user/competition2'
EXP_DIR = '/home/user/competition2/experiments/exp16_diverse_ensemble'

print("=" * 60)
print("Exp16: Diverse Ensemble (LGB + XGB + CAT + ET + RF)")
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
print(f"Seeds: {N_SEEDS}")

X_train = train_df[features]
y_train = target
X_test = test_df[features]

# For tree-based sklearn models, fill NaN
X_train_filled = X_train.fillna(-999)
X_test_filled = X_test.fillna(-999)

# Initialize OOF and predictions for each model
oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_et = np.zeros(len(train_df))
oof_rf = np.zeros(len(train_df))

pred_lgb = np.zeros(len(test_df))
pred_xgb = np.zeros(len(test_df))
pred_cat = np.zeros(len(test_df))
pred_et = np.zeros(len(test_df))
pred_rf = np.zeros(len(test_df))

print("\nTraining 5 models x 5 seeds...")
for seed_idx, seed in enumerate(SEEDS):
    print(f"  Seed {seed_idx+1}/{N_SEEDS}: {seed}")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
        X_tr_filled = X_train_filled.iloc[tr_idx]
        X_va_filled = X_train_filled.iloc[va_idx]

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

        # ExtraTrees
        model_et = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=seed,
            n_jobs=-1
        )
        model_et.fit(X_tr_filled, y_tr)
        oof_et[va_idx] += model_et.predict_proba(X_va_filled)[:, 1] / N_SEEDS
        pred_et += model_et.predict_proba(X_test_filled)[:, 1] / (N_FOLDS * N_SEEDS)

        # RandomForest
        model_rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=seed,
            n_jobs=-1
        )
        model_rf.fit(X_tr_filled, y_tr)
        oof_rf[va_idx] += model_rf.predict_proba(X_va_filled)[:, 1] / N_SEEDS
        pred_rf += model_rf.predict_proba(X_test_filled)[:, 1] / (N_FOLDS * N_SEEDS)

# Calculate individual CVs
cv_lgb = roc_auc_score(y_train, oof_lgb)
cv_xgb = roc_auc_score(y_train, oof_xgb)
cv_cat = roc_auc_score(y_train, oof_cat)
cv_et = roc_auc_score(y_train, oof_et)
cv_rf = roc_auc_score(y_train, oof_rf)

print("\n" + "=" * 60)
print("Individual Model CVs")
print("=" * 60)
print(f"  LGB: {cv_lgb:.5f}")
print(f"  XGB: {cv_xgb:.5f}")
print(f"  CAT: {cv_cat:.5f}")
print(f"  ET:  {cv_et:.5f}")
print(f"  RF:  {cv_rf:.5f}")

# Try different ensemble combinations
print("\n" + "=" * 60)
print("Ensemble Combinations")
print("=" * 60)

# Original 3-model ensemble
oof_3model = oof_lgb * 0.3 + oof_xgb * 0.3 + oof_cat * 0.4
pred_3model = pred_lgb * 0.3 + pred_xgb * 0.3 + pred_cat * 0.4
cv_3model = roc_auc_score(y_train, oof_3model)
print(f"3-Model (LGB+XGB+CAT): {cv_3model:.5f}")

# 5-model equal weights
oof_5model_eq = (oof_lgb + oof_xgb + oof_cat + oof_et + oof_rf) / 5
pred_5model_eq = (pred_lgb + pred_xgb + pred_cat + pred_et + pred_rf) / 5
cv_5model_eq = roc_auc_score(y_train, oof_5model_eq)
print(f"5-Model (equal): {cv_5model_eq:.5f}")

# 5-model weighted (favor GBDT)
oof_5model_w1 = oof_lgb * 0.25 + oof_xgb * 0.25 + oof_cat * 0.3 + oof_et * 0.1 + oof_rf * 0.1
pred_5model_w1 = pred_lgb * 0.25 + pred_xgb * 0.25 + pred_cat * 0.3 + pred_et * 0.1 + pred_rf * 0.1
cv_5model_w1 = roc_auc_score(y_train, oof_5model_w1)
print(f"5-Model (GBDT heavy): {cv_5model_w1:.5f}")

# Optimize weights
from scipy.optimize import minimize

def neg_auc(weights):
    w = weights / weights.sum()
    oof = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat + w[3]*oof_et + w[4]*oof_rf
    return -roc_auc_score(y_train, oof)

result = minimize(neg_auc, x0=[0.2, 0.2, 0.2, 0.2, 0.2],
                  bounds=[(0, 1)]*5, method='SLSQP')
best_weights = result.x / result.x.sum()

oof_5model_opt = (best_weights[0]*oof_lgb + best_weights[1]*oof_xgb +
                  best_weights[2]*oof_cat + best_weights[3]*oof_et + best_weights[4]*oof_rf)
pred_5model_opt = (best_weights[0]*pred_lgb + best_weights[1]*pred_xgb +
                   best_weights[2]*pred_cat + best_weights[3]*pred_et + best_weights[4]*pred_rf)
cv_5model_opt = roc_auc_score(y_train, oof_5model_opt)
print(f"5-Model (optimized): {cv_5model_opt:.5f}")
print(f"  Weights: LGB={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, CAT={best_weights[2]:.3f}, ET={best_weights[3]:.3f}, RF={best_weights[4]:.3f}")

# Find best combination
results = {
    '3-Model': (cv_3model, pred_3model),
    '5-Model-equal': (cv_5model_eq, pred_5model_eq),
    '5-Model-GBDT': (cv_5model_w1, pred_5model_w1),
    '5-Model-opt': (cv_5model_opt, pred_5model_opt)
}

best_name = max(results, key=lambda x: results[x][0])
best_cv = results[best_name][0]
best_pred = results[best_name][1]

print(f"\nBest: {best_name} with CV = {best_cv:.5f}")

# Save all submissions
for name, (cv, pred) in results.items():
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'Drafted': pred
    })
    submission.to_csv(f'{EXP_DIR}/submission_{name.replace("-", "_")}.csv', index=False)

# Save best as main submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': best_pred
})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

# Save OOF
np.save(f'{EXP_DIR}/oof_lgb.npy', oof_lgb)
np.save(f'{EXP_DIR}/oof_xgb.npy', oof_xgb)
np.save(f'{EXP_DIR}/oof_cat.npy', oof_cat)
np.save(f'{EXP_DIR}/oof_et.npy', oof_et)
np.save(f'{EXP_DIR}/oof_rf.npy', oof_rf)

# Save results
results_summary = {
    'cv_lgb': cv_lgb,
    'cv_xgb': cv_xgb,
    'cv_cat': cv_cat,
    'cv_et': cv_et,
    'cv_rf': cv_rf,
    'cv_3model': cv_3model,
    'cv_5model_equal': cv_5model_eq,
    'cv_5model_gbdt': cv_5model_w1,
    'cv_5model_opt': cv_5model_opt,
    'best_weights': best_weights.tolist(),
    'best_ensemble': best_name,
    'best_cv': best_cv,
    'exp13_cv': 0.85138,
    'exp13_lb': 0.84524
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to {EXP_DIR}")
print(f"\nExp13 reference: CV=0.85138, LB=0.84524")
