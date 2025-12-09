"""
Exp27: Feature Selection using Chatterjee Correlation
- Compare Chatterjee-based vs Importance-based feature selection
- Test composite score (Chatterjee + Pearson + Spearman) for selection
- Validate with cross-validation
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
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

EXP_DIR = '/home/user/competition2/experiments/exp27_chatterjee_correlation'
BASE_DIR = '/home/user/competition2'

print("=" * 60)
print("Exp27: Chatterjee-based Feature Selection Validation")
print("=" * 60)

# ==========================================
# Load correlation analysis results
# ==========================================
with open(f'{EXP_DIR}/analysis_results.json', 'r') as f:
    analysis_results = json.load(f)

correlation_df = pd.read_csv(f'{EXP_DIR}/correlation_comparison.csv')

# ==========================================
# Chatterjee Correlation Implementation
# ==========================================
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

    exclude_cols = ['Id', 'Drafted', 'is_train']
    features = [c for c in train_df.columns if c not in exclude_cols]

    return train_df, test_df, target, features, cat_cols


# ==========================================
# Cross-Validation Function
# ==========================================
def run_cv(train_df, target, features, n_folds=5, n_seeds=3):
    """Run 3-model ensemble CV with multiple seeds"""
    seeds = [42, 2023, 101][:n_seeds]
    cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']
    cat_features = [f for f in features if f in cat_cols]

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
        'n_jobs': -1
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.02,
        'n_estimators': 500,
        'max_depth': 6,
        'min_child_weight': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': 0,
        'n_jobs': -1
    }

    all_oof = []

    for seed in seeds:
        oof_lgb = np.zeros(len(train_df))
        oof_xgb = np.zeros(len(train_df))
        oof_cat = np.zeros(len(train_df))

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        X = train_df[features].values
        y = target.values

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # LightGBM
            lgb_p = lgb_params.copy()
            lgb_p['random_state'] = seed
            model_lgb = lgb.LGBMClassifier(**lgb_p)
            model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            oof_lgb[va_idx] = model_lgb.predict_proba(X_va)[:, 1]

            # XGBoost
            xgb_p = xgb_params.copy()
            xgb_p['random_state'] = seed
            model_xgb = xgb.XGBClassifier(**xgb_p)
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_xgb[va_idx] = model_xgb.predict_proba(X_va)[:, 1]

            # CatBoost
            cat_idx = [features.index(c) for c in cat_features if c in features]
            model_cat = CatBoostClassifier(
                iterations=500,
                learning_rate=0.02,
                depth=6,
                l2_leaf_reg=3,
                random_seed=seed,
                verbose=False,
            )
            # Prepare data for CatBoost
            train_pool_df = pd.DataFrame(X_tr, columns=features)
            val_pool_df = pd.DataFrame(X_va, columns=features)
            if cat_idx:
                for idx in cat_idx:
                    col = features[idx]
                    train_pool_df[col] = train_pool_df[col].astype(int).astype(str)
                    val_pool_df[col] = val_pool_df[col].astype(int).astype(str)
                model_cat.fit(train_pool_df, y_tr, eval_set=(val_pool_df, y_va), cat_features=cat_idx)
                oof_cat[va_idx] = model_cat.predict_proba(val_pool_df)[:, 1]
            else:
                model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
                oof_cat[va_idx] = model_cat.predict_proba(X_va)[:, 1]

        # Ensemble
        oof_ensemble = 0.3 * oof_lgb + 0.3 * oof_xgb + 0.4 * oof_cat
        all_oof.append(oof_ensemble)

    # Average across seeds
    final_oof = np.mean(all_oof, axis=0)
    cv_score = roc_auc_score(target, final_oof)

    return cv_score


# ==========================================
# Load data
# ==========================================
print("\nLoading data...")
train_df, test_df, target, all_features, cat_cols = get_data()
print(f"Total features: {len(all_features)}")

# ==========================================
# Feature Selection Strategies
# ==========================================

# Strategy 1: Current exp13 Top30 (Importance-based)
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_importance = [f for f in exp13_results['best_features'] if f in all_features]

# Strategy 2: Top30 by Chatterjee correlation
top30_chatterjee = correlation_df.nlargest(30, 'chatterjee')['feature'].tolist()
top30_chatterjee = [f for f in top30_chatterjee if f in all_features]

# Strategy 3: Top30 by Composite Score
correlation_df['composite'] = (
    0.4 * correlation_df['chatterjee'] +
    0.3 * correlation_df['spearman'] +
    0.3 * correlation_df['pearson']
)
top30_composite = correlation_df.nlargest(30, 'composite')['feature'].tolist()
top30_composite = [f for f in top30_composite if f in all_features]

# Strategy 4: Hybrid - Importance Top20 + Chatterjee Top10 (excluding redundant)
# Remove highly correlated features
chatterjee_candidates = correlation_df[~correlation_df['feature'].isin(top30_importance[:20])]
chatterjee_add = chatterjee_candidates.nlargest(15, 'chatterjee')['feature'].tolist()

# Remove redundant (high pair-wise correlation)
redundant_pairs = [
    ('Sprint_40yd_Pos_Z', 'Sprint_40yd_Pos_Rank'),
    ('Sprint_40yd_Pos_Z', 'Sprint_40yd_Pos_Diff'),
    ('Speed_Score_Pos_Z', 'Speed_Score_Pos_Diff'),
    ('Speed_Score_Pos_Z', 'Speed_Score_Pos_Rank'),
]
to_remove = set()
for f1, f2 in redundant_pairs:
    if f1 in chatterjee_add and f2 in chatterjee_add:
        to_remove.add(f2)  # Keep first, remove second

chatterjee_add_filtered = [f for f in chatterjee_add if f not in to_remove][:10]
top30_hybrid = top30_importance[:20] + [f for f in chatterjee_add_filtered if f in all_features]

# Strategy 5: Non-linear features focus
# Features with high (Chatterjee - Pearson) difference
correlation_df['nonlinear_score'] = correlation_df['chatterjee'] - correlation_df['pearson']
nonlinear_features = correlation_df.nlargest(15, 'nonlinear_score')['feature'].tolist()
# Combine with top importance features
linear_features = correlation_df.nlargest(15, 'pearson')['feature'].tolist()
top30_mixed = list(set(nonlinear_features + linear_features))[:30]
top30_mixed = [f for f in top30_mixed if f in all_features]

# Add categorical features to all strategies
for strategy_features in [top30_chatterjee, top30_composite, top30_hybrid, top30_mixed]:
    for cat in cat_cols:
        if cat not in strategy_features and cat in all_features:
            strategy_features.append(cat)

print("\n" + "=" * 60)
print("Feature Selection Strategies Comparison")
print("=" * 60)

strategies = {
    'exp13_top30_importance': top30_importance,
    'top30_chatterjee': top30_chatterjee,
    'top30_composite': top30_composite,
    'top30_hybrid_imp20_chat10': top30_hybrid,
    'top30_mixed_linear_nonlinear': top30_mixed
}

for name, features in strategies.items():
    print(f"\n{name}: {len(features)} features")
    # Show overlap with exp13
    if name != 'exp13_top30_importance':
        overlap = len(set(features) & set(top30_importance))
        unique = set(features) - set(top30_importance)
        print(f"  Overlap with exp13: {overlap}")
        print(f"  Unique features: {sorted(unique)[:5]}...")

# ==========================================
# Run CV for each strategy
# ==========================================
print("\n" + "=" * 60)
print("Cross-Validation Results")
print("=" * 60)
print("(3-model ensemble, 5-fold CV, 3 seeds)")

results = {}
for name, features in strategies.items():
    print(f"\nEvaluating: {name}...")
    valid_features = [f for f in features if f in train_df.columns]
    if len(valid_features) < 10:
        print(f"  Skipping: too few valid features ({len(valid_features)})")
        continue

    cv_score = run_cv(train_df, target, valid_features, n_folds=5, n_seeds=3)
    results[name] = {
        'cv_auc': cv_score,
        'n_features': len(valid_features),
        'features': valid_features
    }
    print(f"  CV AUC: {cv_score:.5f} ({len(valid_features)} features)")

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_auc'], reverse=True)
print(f"\n{'Strategy':40} {'CV AUC':>10} {'N Features':>12}")
print("-" * 65)
for name, res in sorted_results:
    print(f"{name:40} {res['cv_auc']:10.5f} {res['n_features']:12}")

best_strategy = sorted_results[0][0]
best_score = sorted_results[0][1]['cv_auc']
baseline_score = results.get('exp13_top30_importance', {}).get('cv_auc', 0)

print(f"\nBest strategy: {best_strategy}")
print(f"Best CV AUC: {best_score:.5f}")
if baseline_score > 0:
    improvement = best_score - baseline_score
    print(f"Improvement over baseline: {improvement:+.5f}")

# Save results
with open(f'{EXP_DIR}/cv_results.json', 'w') as f:
    # Convert to serializable format
    save_results = {k: {'cv_auc': v['cv_auc'], 'n_features': v['n_features']}
                    for k, v in results.items()}
    json.dump(save_results, f, indent=2)

print(f"\nResults saved to {EXP_DIR}/cv_results.json")
