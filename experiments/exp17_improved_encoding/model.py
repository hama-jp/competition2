"""
Exp17: Improved Target Encoding
- Leave-One-Out encoding
- Bayesian smoothing with different parameters
- Compare different encoding strategies
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

BASE_DIR = '/home/user/competition2'
EXP_DIR = '/home/user/competition2/experiments/exp17_improved_encoding'

print("=" * 60)
print("Exp17: Improved Target Encoding")
print("=" * 60)

# ==========================================
# Different Target Encoding Methods
# ==========================================

def target_encode_kfold(train_df, test_df, col, target, n_folds=5, smoothing=10):
    """Original K-Fold Target Encoding"""
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

def target_encode_loo(train_df, test_df, col, target, smoothing=10):
    """Leave-One-Out Target Encoding - more robust for train data"""
    global_mean = target.mean()

    # For training: leave-one-out
    train_encoded = np.zeros(len(train_df))

    # Calculate sum and count for each category
    cat_sum = train_df.groupby(col).apply(lambda x: target.loc[x.index].sum())
    cat_count = train_df.groupby(col).apply(lambda x: len(x))

    for idx in range(len(train_df)):
        cat = train_df.iloc[idx][col]
        cat_s = cat_sum.get(cat, 0)
        cat_c = cat_count.get(cat, 0)
        y_i = target.iloc[idx]

        # Leave this sample out
        loo_sum = cat_s - y_i
        loo_count = cat_c - 1

        if loo_count > 0:
            train_encoded[idx] = (loo_sum + smoothing * global_mean) / (loo_count + smoothing)
        else:
            train_encoded[idx] = global_mean

    # For test: use full training data
    agg_full = train_df.groupby(col).apply(lambda x: (
        (target.loc[x.index].sum() + smoothing * global_mean) /
        (len(x) + smoothing)
    ))
    test_encoded = test_df[col].map(agg_full).fillna(global_mean).values

    return train_encoded, test_encoded

def target_encode_bayesian(train_df, test_df, col, target, prior_weight=10):
    """Bayesian Target Encoding with credible intervals"""
    global_mean = target.mean()
    global_var = target.var()

    # For training: K-fold to prevent leakage
    train_encoded = np.zeros(len(train_df))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(train_df, target):
        tr_target = target.iloc[tr_idx]
        tr_data = train_df.iloc[tr_idx]

        cat_stats = tr_data.groupby(col).apply(lambda x: pd.Series({
            'sum': tr_target.loc[x.index].sum(),
            'count': len(x),
            'var': tr_target.loc[x.index].var() if len(x) > 1 else global_var
        }))

        for va_i in va_idx:
            cat = train_df.iloc[va_i][col]
            if cat in cat_stats.index:
                n = cat_stats.loc[cat, 'count']
                cat_mean = cat_stats.loc[cat, 'sum'] / n
                # Bayesian shrinkage
                weight = n / (n + prior_weight)
                train_encoded[va_i] = weight * cat_mean + (1 - weight) * global_mean
            else:
                train_encoded[va_i] = global_mean

    # For test: use full training data
    cat_stats = train_df.groupby(col).apply(lambda x: pd.Series({
        'sum': target.loc[x.index].sum(),
        'count': len(x)
    }))

    test_encoded = np.zeros(len(test_df))
    for i in range(len(test_df)):
        cat = test_df.iloc[i][col]
        if cat in cat_stats.index:
            n = cat_stats.loc[cat, 'count']
            cat_mean = cat_stats.loc[cat, 'sum'] / n
            weight = n / (n + prior_weight)
            test_encoded[i] = weight * cat_mean + (1 - weight) * global_mean
        else:
            test_encoded[i] = global_mean

    return train_encoded, test_encoded

# ==========================================
# Data Preparation with encoding options
# ==========================================
def get_data(encoding_method='kfold', smoothing=20):
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

    # Apply encoding based on method
    if encoding_method == 'kfold':
        train_df['School_TE'], test_df['School_TE'] = target_encode_kfold(train_temp, test_temp, 'School', target, smoothing=smoothing)
        train_df['Position_TE'], test_df['Position_TE'] = target_encode_kfold(train_temp, test_temp, 'Position', target, smoothing=50)
        train_df['Position_Type_TE'], test_df['Position_Type_TE'] = target_encode_kfold(train_temp, test_temp, 'Position_Type', target, smoothing=100)
    elif encoding_method == 'loo':
        train_df['School_TE'], test_df['School_TE'] = target_encode_loo(train_temp, test_temp, 'School', target, smoothing=smoothing)
        train_df['Position_TE'], test_df['Position_TE'] = target_encode_loo(train_temp, test_temp, 'Position', target, smoothing=50)
        train_df['Position_Type_TE'], test_df['Position_Type_TE'] = target_encode_loo(train_temp, test_temp, 'Position_Type', target, smoothing=100)
    elif encoding_method == 'bayesian':
        train_df['School_TE'], test_df['School_TE'] = target_encode_bayesian(train_temp, test_temp, 'School', target, prior_weight=smoothing)
        train_df['Position_TE'], test_df['Position_TE'] = target_encode_bayesian(train_temp, test_temp, 'Position', target, prior_weight=50)
        train_df['Position_Type_TE'], test_df['Position_Type_TE'] = target_encode_bayesian(train_temp, test_temp, 'Position_Type', target, prior_weight=100)

    return train_df, test_df, target

# ==========================================
# Training function
# ==========================================
def train_and_evaluate(train_df, test_df, target, features, cat_cols):
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    X_train = train_df[features]
    y_train = target
    X_test = test_df[features]

    # Load best params
    with open('/home/user/competition2/experiments/exp07_final/results.json', 'r') as f:
        exp07_results = json.load(f)
    lgb_params = exp07_results['best_params_lgb']
    xgb_params = exp07_results['best_params_xgb']
    cat_params = exp07_results['best_params_cat']

    oof = np.zeros(len(train_df))
    pred = np.zeros(len(test_df))

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

            # XGB
            xgb_p = xgb_params.copy()
            xgb_p['random_state'] = seed
            model_xgb = xgb.XGBClassifier(**xgb_p, n_estimators=10000)
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            # CatBoost
            cat_p = cat_params.copy()
            cat_p['random_seed'] = seed
            model_cat = CatBoostClassifier(**cat_p)
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)
            model_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

            # Ensemble
            pred_va = (model_lgb.predict_proba(X_va)[:, 1] * 0.3 +
                      model_xgb.predict_proba(X_va)[:, 1] * 0.3 +
                      model_cat.predict_proba(X_va)[:, 1] * 0.4)
            pred_test = (model_lgb.predict_proba(X_test)[:, 1] * 0.3 +
                        model_xgb.predict_proba(X_test)[:, 1] * 0.3 +
                        model_cat.predict_proba(X_test)[:, 1] * 0.4)

            oof[va_idx] += pred_va / N_SEEDS
            pred += pred_test / (N_FOLDS * N_SEEDS)

    cv = roc_auc_score(y_train, oof)
    return oof, pred, cv

# ==========================================
# Main
# ==========================================
# Load top 30 features
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top_30_features = exp13_results['best_features']

# Test different encoding methods
encoding_configs = [
    ('kfold', 20, 'K-Fold (baseline)'),
    ('kfold', 30, 'K-Fold smooth=30'),
    ('kfold', 50, 'K-Fold smooth=50'),
    ('loo', 20, 'LOO smooth=20'),
    ('loo', 30, 'LOO smooth=30'),
    ('bayesian', 20, 'Bayesian prior=20'),
    ('bayesian', 30, 'Bayesian prior=30'),
]

results = {}
best_cv = 0
best_pred = None
best_config = None

print("Testing different encoding methods...")
cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']

for method, smoothing, name in encoding_configs:
    print(f"\n--- {name} ---")
    train_df, test_df, target = get_data(encoding_method=method, smoothing=smoothing)
    features = [f for f in top_30_features if f in train_df.columns]

    oof, pred, cv = train_and_evaluate(train_df, test_df, target, features, cat_cols)
    results[name] = {'cv': cv, 'pred': pred}
    print(f"CV: {cv:.5f}")

    if cv > best_cv:
        best_cv = cv
        best_pred = pred
        best_config = name

# Results
print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)

for name, res in results.items():
    marker = " <-- BEST" if name == best_config else ""
    print(f"  {name:25}: CV = {res['cv']:.5f}{marker}")

print(f"\nExp13 (baseline): CV = 0.85138, LB = 0.84524")

# Save submissions
for name, res in results.items():
    safe_name = name.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'Drafted': res['pred']
    })
    submission.to_csv(f'{EXP_DIR}/submission_{safe_name}.csv', index=False)

# Save best as main submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': best_pred
})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

# Save results
results_summary = {
    'encoding_results': {k: v['cv'] for k, v in results.items()},
    'best_config': best_config,
    'best_cv': best_cv,
    'exp13_cv': 0.85138,
    'exp13_lb': 0.84524
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to {EXP_DIR}")
