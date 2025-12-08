"""
Exp25b: Balanced Pseudo-labeling
- Adjust pseudo-label balance to match training data (64.8% pos / 35.2% neg)
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
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

EXP_DIR = '/home/user/competition2/experiments/exp25_improved_pseudo'
BASE_DIR = '/home/user/competition2'

print("=" * 60)
print("Exp25b: Balanced Pseudo-labeling")
print("=" * 60)

# Load exp13 results
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']

# 30 features (best from exp24)
features_31 = top30_features + ['Agility_3cone_Pos_Diff']
features = [f for f in features_31 if f != 'Bench_per_Weight']
print(f"Features: {len(features)}")

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
# Load data
# ==========================================
print("\nLoading data...")
train_df, test_df, target, cat_cols = get_data()

X_train = train_df[features]
y_train = target
X_test = test_df[features]

cat_indices = [features.index(c) for c in cat_cols if c in features]

# Training data balance
train_pos_ratio = target.mean()
print(f"Training balance: {train_pos_ratio*100:.1f}% positive / {(1-train_pos_ratio)*100:.1f}% negative")

# ==========================================
# Get predictions from each model
# ==========================================
print("\n--- Getting model predictions ---")

pred_lgb = np.zeros(len(test_df))
pred_xgb = np.zeros(len(test_df))
pred_cat = np.zeros(len(test_df))

for seed in SEEDS[:3]:
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        lgb_p = lgb_params.copy()
        lgb_p['random_state'] = seed
        model_lgb = lgb.LGBMClassifier(**lgb_p, n_estimators=10000)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
        pred_lgb += model_lgb.predict_proba(X_test)[:, 1] / (N_FOLDS * 3)

        xgb_p = xgb_params.copy()
        xgb_p['random_state'] = seed
        model_xgb = xgb.XGBClassifier(**xgb_p, n_estimators=10000)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred_xgb += model_xgb.predict_proba(X_test)[:, 1] / (N_FOLDS * 3)

        cat_p = cat_params.copy()
        cat_p['random_seed'] = seed
        model_cat = CatBoostClassifier(**cat_p)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
        pred_cat += model_cat.predict_proba(X_test)[:, 1] / (N_FOLDS * 3)

pred_ensemble = pred_lgb * 0.3 + pred_xgb * 0.3 + pred_cat * 0.4

# ==========================================
# Generate balanced pseudo-labels
# ==========================================
print("\n" + "=" * 60)
print("Balanced Pseudo-labeling")
print("=" * 60)

# Get high confidence samples (threshold 0.35)
threshold = 0.35
confidence = np.abs(pred_ensemble - 0.5)
high_conf_mask = confidence >= threshold
labels = (pred_ensemble >= 0.5).astype(int)

# Get indices for positive and negative
pos_indices = np.where(high_conf_mask & (labels == 1))[0]
neg_indices = np.where(high_conf_mask & (labels == 0))[0]

print(f"Original high-confidence samples:")
print(f"  Positive: {len(pos_indices)} ({len(pos_indices)/(len(pos_indices)+len(neg_indices))*100:.1f}%)")
print(f"  Negative: {len(neg_indices)} ({len(neg_indices)/(len(pos_indices)+len(neg_indices))*100:.1f}%)")

# Balance to match training ratio (64.8% positive)
# Keep all positives, subsample negatives
target_neg_count = int(len(pos_indices) * (1 - train_pos_ratio) / train_pos_ratio)
np.random.seed(42)

if target_neg_count < len(neg_indices):
    # Subsample negatives (keep highest confidence ones)
    neg_confidence = confidence[neg_indices]
    top_neg_indices = neg_indices[np.argsort(-neg_confidence)[:target_neg_count]]
    balanced_neg_indices = top_neg_indices
else:
    balanced_neg_indices = neg_indices

print(f"\nBalanced pseudo-labels (matching train ratio {train_pos_ratio*100:.1f}%):")
print(f"  Positive: {len(pos_indices)}")
print(f"  Negative: {len(balanced_neg_indices)}")
total_balanced = len(pos_indices) + len(balanced_neg_indices)
print(f"  Total: {total_balanced}")
print(f"  Ratio: {len(pos_indices)/total_balanced*100:.1f}% / {len(balanced_neg_indices)/total_balanced*100:.1f}%")

# Create balanced mask and labels
balanced_indices = np.concatenate([pos_indices, balanced_neg_indices])
balanced_mask = np.zeros(len(test_df), dtype=bool)
balanced_mask[balanced_indices] = True
balanced_labels = labels.copy()

# ==========================================
# Test with quick CV
# ==========================================
print("\n" + "=" * 60)
print("Testing with Quick CV")
print("=" * 60)

def quick_cv_with_pseudo(X_train_np, y_train_np, X_test_np, pseudo_mask, pseudo_labels, n_seeds=3):
    X_pseudo = X_test_np[pseudo_mask]
    y_pseudo = pseudo_labels[pseudo_mask]

    oof = np.zeros(len(y_train_np))

    for seed in [42, 2023, 101][:n_seeds]:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_np, y_train_np)):
            X_tr = np.vstack([X_train_np[tr_idx], X_pseudo])
            y_tr = np.concatenate([y_train_np[tr_idx], y_pseudo])
            X_va, y_va = X_train_np[va_idx], y_train_np[va_idx]

            lgb_p = lgb_params.copy()
            lgb_p['random_state'] = seed
            lgb_p['n_estimators'] = 10000
            model = lgb.LGBMClassifier(**lgb_p)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va_idx] += model.predict_proba(X_va)[:, 1] / n_seeds

    return roc_auc_score(y_train_np, oof)

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values

# Baseline
print("--- Baseline (no pseudo) ---")
cv_baseline = quick_cv_with_pseudo(X_train_np, y_train_np, X_test_np,
                                    np.zeros(len(test_df), dtype=bool),
                                    np.zeros(len(test_df)))
print(f"CV: {cv_baseline:.5f}")

# Original (unbalanced)
print("\n--- Original (unbalanced, 277 samples) ---")
cv_original = quick_cv_with_pseudo(X_train_np, y_train_np, X_test_np,
                                    high_conf_mask, labels)
print(f"CV: {cv_original:.5f} ({cv_original - cv_baseline:+.5f})")

# Balanced
print(f"\n--- Balanced ({total_balanced} samples) ---")
cv_balanced = quick_cv_with_pseudo(X_train_np, y_train_np, X_test_np,
                                    balanced_mask, balanced_labels)
print(f"CV: {cv_balanced:.5f} ({cv_balanced - cv_baseline:+.5f})")

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Baseline:   CV = {cv_baseline:.5f}")
print(f"Original:   CV = {cv_original:.5f} ({cv_original - cv_baseline:+.5f}) [277 samples, 58.5% pos]")
print(f"Balanced:   CV = {cv_balanced:.5f} ({cv_balanced - cv_baseline:+.5f}) [{total_balanced} samples, {len(pos_indices)/total_balanced*100:.1f}% pos]")

# Save results
results = {
    'baseline_cv': float(cv_baseline),
    'original_cv': float(cv_original),
    'original_samples': 277,
    'original_pos_ratio': 58.5,
    'balanced_cv': float(cv_balanced),
    'balanced_samples': int(total_balanced),
    'balanced_pos_ratio': float(len(pos_indices)/total_balanced*100),
    'train_pos_ratio': float(train_pos_ratio * 100)
}

with open(f'{EXP_DIR}/balanced_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {EXP_DIR}/balanced_results.json")
