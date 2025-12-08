"""
Exp25: Improved Pseudo-labeling
- Higher confidence threshold (0.40, 0.45)
- Multi-model consensus (all 3 models must agree)
- Use 30-feature setup from exp24
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
print("Exp25: Improved Pseudo-labeling")
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
# Feature Engineering (same as exp24)
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

# ==========================================
# Step 1: Train baseline model and get test predictions from each model
# ==========================================
print("\n" + "=" * 60)
print("Step 1: Get predictions from each model")
print("=" * 60)

pred_lgb_all = np.zeros(len(test_df))
pred_xgb_all = np.zeros(len(test_df))
pred_cat_all = np.zeros(len(test_df))

for seed_idx, seed in enumerate(SEEDS[:3]):  # Use 3 seeds for speed
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
        pred_lgb_all += model_lgb.predict_proba(X_test)[:, 1] / (N_FOLDS * 3)

        # XGB
        xgb_p = xgb_params.copy()
        xgb_p['random_state'] = seed
        model_xgb = xgb.XGBClassifier(**xgb_p, n_estimators=10000)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred_xgb_all += model_xgb.predict_proba(X_test)[:, 1] / (N_FOLDS * 3)

        # CatBoost
        cat_p = cat_params.copy()
        cat_p['random_seed'] = seed
        model_cat = CatBoostClassifier(**cat_p)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
        pred_cat_all += model_cat.predict_proba(X_test)[:, 1] / (N_FOLDS * 3)

print(f"LGB predictions: mean={pred_lgb_all.mean():.4f}")
print(f"XGB predictions: mean={pred_xgb_all.mean():.4f}")
print(f"CAT predictions: mean={pred_cat_all.mean():.4f}")

# ==========================================
# Step 2: Generate pseudo-labels with different strategies
# ==========================================
print("\n" + "=" * 60)
print("Step 2: Generate pseudo-labels")
print("=" * 60)

# Ensemble prediction
pred_ensemble = pred_lgb_all * 0.3 + pred_xgb_all * 0.3 + pred_cat_all * 0.4

# Strategy 1: Single model high threshold (original approach)
def get_pseudo_labels_single(pred, threshold):
    """Get pseudo labels based on single model confidence"""
    confidence = np.abs(pred - 0.5)
    mask = confidence >= threshold
    labels = (pred >= 0.5).astype(int)
    return mask, labels

# Strategy 2: Multi-model consensus
def get_pseudo_labels_consensus(pred_lgb, pred_xgb, pred_cat, threshold):
    """Get pseudo labels only where all models agree with high confidence"""
    # Convert to binary predictions
    label_lgb = (pred_lgb >= 0.5).astype(int)
    label_xgb = (pred_xgb >= 0.5).astype(int)
    label_cat = (pred_cat >= 0.5).astype(int)

    # Check consensus
    consensus = (label_lgb == label_xgb) & (label_xgb == label_cat)

    # Check confidence for each model
    conf_lgb = np.abs(pred_lgb - 0.5) >= threshold
    conf_xgb = np.abs(pred_xgb - 0.5) >= threshold
    conf_cat = np.abs(pred_cat - 0.5) >= threshold

    # All models must have high confidence AND agree
    mask = consensus & conf_lgb & conf_xgb & conf_cat
    labels = label_lgb  # All same, pick any

    return mask, labels

# Test different strategies
strategies = []

# Original (threshold 0.35)
mask, labels = get_pseudo_labels_single(pred_ensemble, 0.35)
strategies.append({
    'name': 'single_0.35',
    'mask': mask,
    'labels': labels,
    'n_samples': mask.sum(),
    'n_pos': labels[mask].sum(),
    'n_neg': (1 - labels[mask]).sum()
})

# Higher thresholds
for thresh in [0.40, 0.45]:
    mask, labels = get_pseudo_labels_single(pred_ensemble, thresh)
    strategies.append({
        'name': f'single_{thresh}',
        'mask': mask,
        'labels': labels,
        'n_samples': mask.sum(),
        'n_pos': labels[mask].sum(),
        'n_neg': (1 - labels[mask]).sum()
    })

# Multi-model consensus
for thresh in [0.30, 0.35, 0.40]:
    mask, labels = get_pseudo_labels_consensus(pred_lgb_all, pred_xgb_all, pred_cat_all, thresh)
    strategies.append({
        'name': f'consensus_{thresh}',
        'mask': mask,
        'labels': labels,
        'n_samples': mask.sum(),
        'n_pos': labels[mask].sum(),
        'n_neg': (1 - labels[mask]).sum()
    })

print("\nPseudo-labeling strategies:")
print(f"{'Strategy':20} {'Samples':>8} {'Positive':>8} {'Negative':>8}")
print("-" * 50)
for s in strategies:
    print(f"{s['name']:20} {s['n_samples']:8} {s['n_pos']:8} {s['n_neg']:8}")

# ==========================================
# Step 3: Test each strategy with quick CV
# ==========================================
print("\n" + "=" * 60)
print("Step 3: Test strategies with quick CV")
print("=" * 60)

def quick_cv_with_pseudo(train_df, test_df, target, features, pseudo_mask, pseudo_labels, cat_indices, n_seeds=3):
    """Quick CV with pseudo-labeling"""
    X_train = train_df[features].values
    y_train = target.values
    X_test = test_df[features].values

    # Create pseudo-labeled test samples
    X_pseudo = X_test[pseudo_mask]
    y_pseudo = pseudo_labels[pseudo_mask]

    oof = np.zeros(len(train_df))

    for seed in [42, 2023, 101][:n_seeds]:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr_orig = X_train[tr_idx]
            y_tr_orig = y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

            # Add pseudo-labeled samples
            X_tr = np.vstack([X_tr_orig, X_pseudo])
            y_tr = np.concatenate([y_tr_orig, y_pseudo])

            # LGB only for speed
            lgb_p = lgb_params.copy()
            lgb_p['random_state'] = seed
            lgb_p['n_estimators'] = 10000
            model = lgb.LGBMClassifier(**lgb_p)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va_idx] += model.predict_proba(X_va)[:, 1] / n_seeds

    return roc_auc_score(y_train, oof)

# Baseline (no pseudo-labeling)
print("\n--- Baseline (no pseudo-labeling) ---")
cv_baseline = quick_cv_with_pseudo(train_df, test_df, target, features,
                                    np.zeros(len(test_df), dtype=bool),
                                    np.zeros(len(test_df)), cat_indices)
print(f"CV: {cv_baseline:.5f}")

# Test each strategy
results = {'baseline': cv_baseline}

for s in strategies:
    if s['n_samples'] > 0:
        cv = quick_cv_with_pseudo(train_df, test_df, target, features,
                                   s['mask'], s['labels'], cat_indices)
        diff = cv - cv_baseline
        print(f"{s['name']:20}: CV = {cv:.5f} ({'+' if diff >= 0 else ''}{diff:.5f}) [{s['n_samples']} samples]")
        results[s['name']] = cv
        s['cv'] = cv
    else:
        print(f"{s['name']:20}: No samples selected")
        s['cv'] = None

# Find best strategy
valid_strategies = [s for s in strategies if s['cv'] is not None]
if valid_strategies:
    best_strategy = max(valid_strategies, key=lambda x: x['cv'])
    print(f"\nBest strategy: {best_strategy['name']} (CV = {best_strategy['cv']:.5f})")

# Save results
results_summary = {
    'baseline_cv': cv_baseline,
    'strategies': [{k: v for k, v in s.items() if k != 'mask' and k != 'labels'} for s in strategies],
    'best_strategy': best_strategy['name'] if valid_strategies else None,
    'best_cv': best_strategy['cv'] if valid_strategies else None
}

with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to {EXP_DIR}/results.json")
