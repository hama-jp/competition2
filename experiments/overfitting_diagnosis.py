"""
Overfitting Diagnosis: Compare experiments with known LB scores
Analyze metrics that correlate with LB performance
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

BASE_DIR = '/home/user/competition2'
N_FOLDS = 5
SEEDS = [42, 2023, 101]  # 3 seeds for faster diagnosis

# Known LB scores
EXPERIMENTS = {
    'exp21_ensemble': {
        'cv': 0.85161, 'lb': 0.84582,
        'model': 'ensemble',
        'params': None  # Will use ensemble
    },
    'exp33_catboost_default': {
        'cv': 0.85083, 'lb': 0.85130,
        'model': 'catboost',
        'params': {
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
    },
    'exp37_catboost_wide_tune': {
        'cv': 0.85307, 'lb': 0.84655,
        'model': 'catboost',
        'params': {
            'learning_rate': 0.0806817268704256,
            'depth': 3,
            'l2_leaf_reg': 1.0930907772710212,
            'min_data_in_leaf': 82,
            'random_strength': 1.1308911791898697,
            'bagging_temperature': 0.016824602509514452,
            'border_count': 110,
            'iterations': 10000,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False,
            'allow_writing_files': False,
        }
    },
}

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
# Diagnosis function
# ==========================================
def diagnose_model(name, params, X_train, y_train, cat_indices):
    print(f"\n{'='*60}")
    print(f"Diagnosing: {name}")
    print(f"{'='*60}")

    results = {
        'name': name,
        'known_cv': EXPERIMENTS[name]['cv'],
        'known_lb': EXPERIMENTS[name]['lb'],
        'train_aucs': [],
        'val_aucs': [],
        'train_val_gaps': [],
        'early_stop_iters': [],
        'fold_val_aucs': [],
        'seed_cvs': [],
        'predictions': [],
    }

    all_oof = np.zeros(len(X_train))

    for seed in SEEDS:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X_train))
        seed_train_aucs = []
        seed_val_aucs = []
        seed_iters = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

            p = params.copy()
            p['random_seed'] = seed
            model = CatBoostClassifier(**p)
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

            # Train AUC
            train_pred = model.predict_proba(X_tr)[:, 1]
            train_auc = roc_auc_score(y_tr, train_pred)

            # Val AUC
            val_pred = model.predict_proba(X_va)[:, 1]
            val_auc = roc_auc_score(y_va, val_pred)

            oof[va_idx] = val_pred

            # Early stopping iterations
            best_iter = model.get_best_iteration()

            seed_train_aucs.append(train_auc)
            seed_val_aucs.append(val_auc)
            seed_iters.append(best_iter)

            results['fold_val_aucs'].append(val_auc)

        results['train_aucs'].extend(seed_train_aucs)
        results['val_aucs'].extend(seed_val_aucs)
        results['train_val_gaps'].extend([t - v for t, v in zip(seed_train_aucs, seed_val_aucs)])
        results['early_stop_iters'].extend(seed_iters)

        seed_cv = roc_auc_score(y_train, oof)
        results['seed_cvs'].append(seed_cv)
        all_oof += oof / len(SEEDS)

    results['predictions'] = all_oof

    # Calculate summary metrics
    results['mean_train_auc'] = np.mean(results['train_aucs'])
    results['mean_val_auc'] = np.mean(results['val_aucs'])
    results['mean_gap'] = np.mean(results['train_val_gaps'])
    results['std_gap'] = np.std(results['train_val_gaps'])
    results['mean_iter'] = np.mean(results['early_stop_iters'])
    results['std_iter'] = np.std(results['early_stop_iters'])
    results['cv_variance'] = np.std(results['seed_cvs'])
    results['final_cv'] = roc_auc_score(y_train, all_oof)

    # Prediction distribution
    preds = results['predictions']
    results['pred_mean'] = np.mean(preds)
    results['pred_std'] = np.std(preds)
    results['pred_min'] = np.min(preds)
    results['pred_max'] = np.max(preds)
    results['pred_extreme_low'] = np.sum(preds < 0.1) / len(preds)  # % < 0.1
    results['pred_extreme_high'] = np.sum(preds > 0.9) / len(preds)  # % > 0.9

    # Print results
    print(f"\n--- Train vs Val Gap ---")
    print(f"  Mean Train AUC: {results['mean_train_auc']:.5f}")
    print(f"  Mean Val AUC:   {results['mean_val_auc']:.5f}")
    print(f"  Mean Gap:       {results['mean_gap']:.5f} (±{results['std_gap']:.5f})")

    print(f"\n--- Early Stopping ---")
    print(f"  Mean Iterations: {results['mean_iter']:.1f} (±{results['std_iter']:.1f})")

    print(f"\n--- CV Stability ---")
    print(f"  Seed CVs: {[f'{cv:.5f}' for cv in results['seed_cvs']]}")
    print(f"  CV Std:   {results['cv_variance']:.5f}")
    print(f"  Final CV: {results['final_cv']:.5f}")

    print(f"\n--- Prediction Distribution ---")
    print(f"  Mean: {results['pred_mean']:.4f}, Std: {results['pred_std']:.4f}")
    print(f"  Range: [{results['pred_min']:.4f}, {results['pred_max']:.4f}]")
    print(f"  Extreme low (<0.1):  {results['pred_extreme_low']*100:.2f}%")
    print(f"  Extreme high (>0.9): {results['pred_extreme_high']*100:.2f}%")

    print(f"\n--- Known Scores ---")
    print(f"  Known CV: {results['known_cv']:.5f}")
    print(f"  Known LB: {results['known_lb']:.5f}")
    print(f"  CV-LB Gap: {results['known_cv'] - results['known_lb']:.5f}")

    return results

# ==========================================
# Main
# ==========================================
print("Loading data...")
train_df, test_df, target, cat_cols = get_data()

# Load features
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top30_features = exp13_results['best_features']
features = top30_features + ['Agility_3cone_Pos_Diff']

X_train = train_df[features]
y_train = target
cat_indices = [features.index(c) for c in cat_cols if c in features]

print(f"Train: {len(train_df)}, Features: {len(features)}")

# Run diagnosis for CatBoost models
all_results = {}
for exp_name, exp_info in EXPERIMENTS.items():
    if exp_info['model'] == 'catboost' and exp_info['params'] is not None:
        results = diagnose_model(exp_name, exp_info['params'], X_train, y_train, cat_indices)
        all_results[exp_name] = results

# ==========================================
# Summary comparison
# ==========================================
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)

print("\n{:<30} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "Experiment", "Train-Val", "Iterations", "CV Std", "Pred Std", "CV-LB"
))
print("-" * 80)

for name, r in all_results.items():
    cv_lb_gap = r['known_cv'] - r['known_lb']
    print("{:<30} {:>10.5f} {:>10.1f} {:>10.5f} {:>10.4f} {:>10.5f}".format(
        name, r['mean_gap'], r['mean_iter'], r['cv_variance'], r['pred_std'], cv_lb_gap
    ))

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS: What predicts good LB performance?")
print("=" * 80)

best_lb_exp = max(all_results.keys(), key=lambda x: all_results[x]['known_lb'])
worst_lb_exp = min(all_results.keys(), key=lambda x: all_results[x]['known_lb'])

best = all_results[best_lb_exp]
worst = all_results[worst_lb_exp]

print(f"\nBest LB: {best_lb_exp} (LB={best['known_lb']:.5f})")
print(f"Worst LB: {worst_lb_exp} (LB={worst['known_lb']:.5f})")

print(f"\nKey differences:")
print(f"  Train-Val Gap: {best['mean_gap']:.5f} vs {worst['mean_gap']:.5f} (diff: {best['mean_gap'] - worst['mean_gap']:+.5f})")
print(f"  Iterations:    {best['mean_iter']:.1f} vs {worst['mean_iter']:.1f} (diff: {best['mean_iter'] - worst['mean_iter']:+.1f})")
print(f"  CV Std:        {best['cv_variance']:.5f} vs {worst['cv_variance']:.5f} (diff: {best['cv_variance'] - worst['cv_variance']:+.5f})")
print(f"  Pred Std:      {best['pred_std']:.4f} vs {worst['pred_std']:.4f} (diff: {best['pred_std'] - worst['pred_std']:+.4f})")

# Save results
with open('/home/user/competition2/experiments/overfitting_diagnosis_results.json', 'w') as f:
    # Convert numpy to python types
    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in r.items()
            if k != 'predictions'  # Skip large array
        }
        save_results[name]['seed_cvs'] = [float(x) for x in r['seed_cvs']]
    json.dump(save_results, f, indent=2)

print(f"\nResults saved to /home/user/competition2/experiments/overfitting_diagnosis_results.json")
