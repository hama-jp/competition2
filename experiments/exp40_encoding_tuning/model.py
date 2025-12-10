"""
Exp40: Target Encoding Tuning
- Base: exp33 (CatBoost default params, 31 features)
- Strategy: Try different smoothing parameters for target encoding
- Higher smoothing = more regularization = potentially more stable
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 5
SEEDS = [42, 2023, 101, 555, 999]
STABILITY_SEEDS = [42, 2023, 101]

EXP_DIR = '/home/user/competition2/experiments/exp40_encoding_tuning'
BASE_DIR = '/home/user/competition2'

CAT_PARAMS = {
    'learning_rate': 0.07718772443488796,
    'depth': 3,
    'l2_leaf_reg': 0.0033458292447738312,
    'subsample': 0.8523245279212943,
    'min_data_in_leaf': 71,
    'random_strength': 1.2032200146196355,
    'iterations': 10000,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'allow_writing_files': False,
}

print("=" * 60)
print("Exp40: Target Encoding Tuning")
print("=" * 60)

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

def get_data_with_encoding(school_smooth, position_smooth, postype_smooth):
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

    # Apply encoding with specified smoothing
    train_df['School_TE'], test_df['School_TE'] = target_encode(train_temp, test_temp, 'School', target, smoothing=school_smooth)
    train_df['Position_TE'], test_df['Position_TE'] = target_encode(train_temp, test_temp, 'Position', target, smoothing=position_smooth)
    train_df['Position_Type_TE'], test_df['Position_Type_TE'] = target_encode(train_temp, test_temp, 'Position_Type', target, smoothing=postype_smooth)

    return train_df, test_df, target, cat_cols

# Load features
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']
features = top30_features + ['Agility_3cone_Pos_Diff']

# ==========================================
# Test different smoothing configurations
# ==========================================
print("\n" + "=" * 60)
print("Testing smoothing configurations")
print("=" * 60)

# exp33 baseline: School=20, Position=50, Position_Type=100
configs = [
    {'school': 20, 'position': 50, 'postype': 100, 'name': 'exp33_baseline'},
    {'school': 50, 'position': 100, 'postype': 200, 'name': 'high_smooth'},
    {'school': 100, 'position': 150, 'postype': 300, 'name': 'very_high_smooth'},
    {'school': 30, 'position': 75, 'postype': 150, 'name': 'medium_smooth'},
]

results = {}
for config in configs:
    print(f"\n--- {config['name']} ---")
    print(f"  School={config['school']}, Position={config['position']}, PosType={config['postype']}")

    train_df, test_df, target, cat_cols = get_data_with_encoding(
        config['school'], config['position'], config['postype']
    )

    X_train = train_df[features]
    y_train = target
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    seed_cvs = []
    for seed in STABILITY_SEEDS:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X_train))

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

            p = CAT_PARAMS.copy()
            p['random_seed'] = seed
            model = CatBoostClassifier(**p)
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
            oof[va_idx] = model.predict_proba(X_va)[:, 1]

        seed_cv = roc_auc_score(y_train, oof)
        seed_cvs.append(seed_cv)

    cv_mean = np.mean(seed_cvs)
    cv_std = np.std(seed_cvs)

    results[config['name']] = {
        'config': config,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'seed_cvs': seed_cvs,
    }

    print(f"  CV: {cv_mean:.5f} ± {cv_std:.5f}")
    print(f"  Seed CVs: {[f'{cv:.5f}' for cv in seed_cvs]}")

# ==========================================
# Select best and full training
# ==========================================
print("\n" + "=" * 60)
print("Selecting Best Configuration")
print("=" * 60)

# Select by lowest CV Std
best_name = min(results.keys(), key=lambda x: results[x]['cv_std'])
best = results[best_name]

print(f"\nBest config: {best_name}")
print(f"CV Std: {best['cv_std']:.5f}")

# Show all results
print("\nAll configurations:")
print(f"{'Config':<20} {'CV Mean':<10} {'CV Std':<10}")
print("-" * 40)
for name, r in sorted(results.items(), key=lambda x: x[1]['cv_std']):
    print(f"{name:<20} {r['cv_mean']:.5f}    {r['cv_std']:.5f}")

# Full training with best config
print("\n" + "=" * 60)
print("Full Training (5 seeds)")
print("=" * 60)

best_config = best['config']
train_df, test_df, target, cat_cols = get_data_with_encoding(
    best_config['school'], best_config['position'], best_config['postype']
)

X_train = train_df[features]
y_train = target
X_test = test_df[features]
cat_indices = [features.index(c) for c in cat_cols if c in features]

oof_final = np.zeros(len(train_df))
pred_final = np.zeros(len(test_df))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        p = CAT_PARAMS.copy()
        p['random_seed'] = seed
        model = CatBoostClassifier(**p)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        va_pool = Pool(X_va, y_va, cat_features=cat_indices)
        model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

        oof_final[va_idx] += model.predict_proba(X_va)[:, 1] / len(SEEDS)
        pred_final += model.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

    cv_now = roc_auc_score(y_train, oof_final * len(SEEDS) / (seed_idx + 1))
    print(f"  CV: {cv_now:.5f}")

cv_final = roc_auc_score(y_train, oof_final)

print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"exp40 CV: {cv_final:.5f}")
print(f"exp40 CV Std: {best['cv_std']:.5f}")
print(f"\nexp33: CV=0.85083, CV Std=0.00249, LB=0.85130")
print(f"Difference vs exp33: {cv_final - 0.85083:+.5f}")

# Save
submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': pred_final})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

results_json = {
    'best_config': best_config,
    'cv_final': float(cv_final),
    'cv_std': float(best['cv_std']),
    'all_results': {k: {kk: vv if not isinstance(vv, np.ndarray) else list(vv) for kk, vv in v.items()} for k, v in results.items()},
    'exp33_cv': 0.85083,
    'exp33_lb': 0.85130,
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\nSubmission saved to {EXP_DIR}/submission.csv")
