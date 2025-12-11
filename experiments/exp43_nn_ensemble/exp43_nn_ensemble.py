"""
Exp43: NN + GBDT Ensemble with Logit Average

目的:
- NNとGBDTの確率キャリブレーションの違いを確認
- ロジット平均アンサンブルの効果を検証

NNモデル:
- scikit-learn MLPClassifier (シンプルなMLP)
- 層構成: (128, 64) or (64, 32) など試行
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.special import logit, expit
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

# ==========================================
# Logit Average Functions
# ==========================================
def safe_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return logit(p)

def logit_average(probs_list, weights):
    logits = [safe_logit(p) for p in probs_list]
    weighted_logit = sum(w * l for w, l in zip(weights, logits))
    return expit(weighted_logit)

def simple_average(probs_list, weights):
    return sum(w * p for w, p in zip(weights, probs_list))

# ==========================================
# Data Preparation
# ==========================================
SELECTED_FEATURES = [
    'Age', 'Speed_Score_Pos_Z', 'Age_div_Explosion', 'Speed_Score_Type_Z',
    'Speed_Score_Pos_Diff', 'Sprint_40yd_Pos_Z', 'Speed_Score', 'Age_x_Momentum',
    'Age_x_Speed', 'Momentum_Pos_Z', 'Player_Type', 'Explosion_Score',
    'Agility_3cone_Pos_Diff', 'Work_Rate_Vertical', 'Sprint_40yd_Type_Z', 'Shuttle',
    'Broad_Jump_Pos_Z', 'Weight', 'Sprint_40yd', 'Sprint_40yd_Pos_Diff',
    'Momentum', 'Position', 'Weight_Pos_Z', 'Work_Rate_Vertical_Pos_Diff',
    'School_Count', 'Height', 'BMI_Pos_Z', 'Momentum_Pos_Diff',
    'Agility_3cone_Pos_Z', 'Agility_3cone_Type_Z', 'Vertical_Jump', 'Year',
    'Bench_Press_Reps_Pos_Diff', 'Explosion_Score_Pos_Diff', 'Explosion_Score_Type_Z',
    'Height_Pos_Diff', 'Shuttle_Pos_Z', 'Vertical_Jump_Type_Z', 'Weight_Type_Z',
    'Agility_3cone', 'Bench_Press_Reps', 'BMI_Pos_Diff', 'Work_Rate_Vertical_Pos_Z',
    'Broad_Jump', 'Shuttle_Type_Z', 'Power_Sum', 'Position_Type', 'Missing_Count',
    'Broad_Jump_Type_Z', 'Weight_Pos_Diff', 'Explosion_Score_Pos_Z', 'Shuttle_Pos_Diff', 'School',
    'Elite_Count', 'Red_Flag_Count', 'Talent_Diff'
]

def get_data():
    print("Loading and preprocessing...")
    train = pd.read_csv('../../train.csv')
    test = pd.read_csv('../../test.csv')

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

    data['School_Count'] = data['School'].map(data['School'].value_counts())

    cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = data[col].fillna('Unknown')
        data[col] = le.fit_transform(data[col].astype(str))

    # Elite / Red Flag
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

    cols = ['Id', 'Drafted', 'is_train'] + SELECTED_FEATURES
    existing_cols = [c for c in cols if c in data.columns]
    data = data[existing_cols]

    features = [c for c in data.columns if c not in ['Id', 'Drafted', 'is_train']]
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    train_df = data[data['is_train'] == 1].reset_index(drop=True)
    test_df = data[data['is_train'] == 0].reset_index(drop=True)
    target = train_df['Drafted']

    return train_df, test_df, target, features, cat_indices

# ==========================================
# MLP Training
# ==========================================
def train_mlp(X, y, X_test, features, hidden_layers=(128, 64), alpha=0.001):
    """
    MLPClassifier with StandardScaler
    """
    name = f"MLP{hidden_layers}"
    print(f"Training {name}...")

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx][features].copy(), y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx][features].copy(), y.iloc[val_idx]
            X_te = X_test[features].copy()

            # Fill NaN with median (NNはNaN非対応)
            for col in features:
                median_val = X_tr[col].median()
                X_tr[col] = X_tr[col].fillna(median_val)
                X_val[col] = X_val[col].fillna(median_val)
                X_te[col] = X_te[col].fillna(median_val)

            # StandardScaler (NNには必須)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            X_te_scaled = scaler.transform(X_te)

            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                alpha=alpha,  # L2 regularization
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed,
                verbose=False
            )

            model.fit(X_tr_scaled, y_tr)

            oof[val_idx] += model.predict_proba(X_val_scaled)[:, 1] / N_SEEDS
            preds += model.predict_proba(X_te_scaled)[:, 1] / (N_FOLDS * N_SEEDS)

    cv_score = roc_auc_score(y, oof)
    print(f"{name} CV: {cv_score:.5f}")
    return oof, preds, cv_score

# ==========================================
# CatBoost Training (Baseline)
# ==========================================
def train_cat(X, y, X_test, features, cat_indices):
    print("Training CatBoost...")
    params = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC',
        'verbose': False, 'allow_writing_files': False,
        'iterations': 5000, 'learning_rate': 0.05, 'depth': 6
    }
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        params['random_seed'] = seed
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx][features], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx][features], y.iloc[val_idx]

            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_val, y_val, cat_features=cat_indices)
            model = CatBoostClassifier(**params)
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

            oof[val_idx] += model.predict_proba(X_val)[:, 1] / N_SEEDS
            preds += model.predict_proba(X_test[features])[:, 1] / (N_FOLDS * N_SEEDS)

    cv_score = roc_auc_score(y, oof)
    print(f"CAT CV: {cv_score:.5f}")
    return oof, preds, cv_score

# ==========================================
# Ensemble Comparison
# ==========================================
def compare_nn_gbdt_ensemble(oof_cat, oof_mlp, target, name_mlp="MLP"):
    """NN + GBDT アンサンブルの効果を検証"""

    print("\n" + "=" * 60)
    print(f"ENSEMBLE: CatBoost + {name_mlp}")
    print("=" * 60)

    # 分布の違いを確認
    print("\n--- Probability Distribution ---")
    print(f"CAT:  mean={np.mean(oof_cat):.4f}, std={np.std(oof_cat):.4f}")
    print(f"{name_mlp}: mean={np.mean(oof_mlp):.4f}, std={np.std(oof_mlp):.4f}")

    # 極端な予測
    print("\n--- Extreme Predictions ---")
    for name, oof in [('CAT', oof_cat), (name_mlp, oof_mlp)]:
        ext_low = np.mean(oof < 0.1) * 100
        ext_high = np.mean(oof > 0.9) * 100
        print(f"{name}: <0.1={ext_low:.1f}%, >0.9={ext_high:.1f}%")

    # 相関
    corr = np.corrcoef(oof_cat, oof_mlp)[0, 1]
    print(f"\nCorrelation (CAT vs {name_mlp}): {corr:.4f}")

    # Weight grid search
    print("\n--- Weight Optimization ---")
    best_simple = {'score': 0, 'w': 0}
    best_logit = {'score': 0, 'w': 0}

    for w_cat in np.arange(0, 1.05, 0.05):
        w_mlp = 1.0 - w_cat

        # Simple average
        simple_pred = w_cat * oof_cat + w_mlp * oof_mlp
        simple_score = roc_auc_score(target, simple_pred)

        # Logit average
        logit_pred = logit_average([oof_cat, oof_mlp], [w_cat, w_mlp])
        logit_score = roc_auc_score(target, logit_pred)

        if simple_score > best_simple['score']:
            best_simple = {'score': simple_score, 'w': w_cat}
        if logit_score > best_logit['score']:
            best_logit = {'score': logit_score, 'w': w_cat}

    print(f"\nBest Simple: CAT={best_simple['w']:.2f}, {name_mlp}={1-best_simple['w']:.2f} -> CV={best_simple['score']:.5f}")
    print(f"Best Logit:  CAT={best_logit['w']:.2f}, {name_mlp}={1-best_logit['w']:.2f} -> CV={best_logit['score']:.5f}")
    print(f"Improvement (Logit - Simple): {best_logit['score'] - best_simple['score']:+.5f}")

    # 固定重みでの比較
    print("\n--- Fixed Weight Comparison ---")
    test_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    print(f"{'w_CAT':<8} {'Simple':>10} {'Logit':>10} {'Diff':>10}")
    print("-" * 40)

    for w_cat in test_weights:
        w_mlp = 1.0 - w_cat
        simple_pred = w_cat * oof_cat + w_mlp * oof_mlp
        logit_pred = logit_average([oof_cat, oof_mlp], [w_cat, w_mlp])
        simple_score = roc_auc_score(target, simple_pred)
        logit_score = roc_auc_score(target, logit_pred)
        diff = logit_score - simple_score
        print(f"{w_cat:<8.1f} {simple_score:>10.5f} {logit_score:>10.5f} {diff:>+10.5f}")

    return best_simple, best_logit

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    train_df, test_df, target, features, cat_indices = get_data()
    print(f"Features: {len(features)}, Train: {len(train_df)}, Test: {len(test_df)}")

    # Train CatBoost
    oof_cat, pred_cat, cv_cat = train_cat(train_df, target, test_df, features, cat_indices)

    # Train various MLP architectures
    results = {}

    # MLP (128, 64) - 標準的
    oof_mlp1, pred_mlp1, cv_mlp1 = train_mlp(train_df, target, test_df, features,
                                              hidden_layers=(128, 64), alpha=0.01)
    results['MLP(128,64)'] = (oof_mlp1, pred_mlp1, cv_mlp1)

    # MLP (64, 32) - 小さめ
    oof_mlp2, pred_mlp2, cv_mlp2 = train_mlp(train_df, target, test_df, features,
                                              hidden_layers=(64, 32), alpha=0.01)
    results['MLP(64,32)'] = (oof_mlp2, pred_mlp2, cv_mlp2)

    # MLP (128, 64, 32) - 3層
    oof_mlp3, pred_mlp3, cv_mlp3 = train_mlp(train_df, target, test_df, features,
                                              hidden_layers=(128, 64, 32), alpha=0.01)
    results['MLP(128,64,32)'] = (oof_mlp3, pred_mlp3, cv_mlp3)

    # MLP with stronger regularization
    oof_mlp4, pred_mlp4, cv_mlp4 = train_mlp(train_df, target, test_df, features,
                                              hidden_layers=(128, 64), alpha=0.1)
    results['MLP(128,64)_reg'] = (oof_mlp4, pred_mlp4, cv_mlp4)

    # Summary
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'CV AUC':>10}")
    print("-" * 30)
    print(f"{'CatBoost':<20} {cv_cat:>10.5f}")
    for name, (_, _, cv) in results.items():
        print(f"{name:<20} {cv:>10.5f}")

    # Compare best MLP with CatBoost
    best_mlp_name = max(results.keys(), key=lambda k: results[k][2])
    best_mlp_oof, best_mlp_pred, _ = results[best_mlp_name]

    print(f"\nBest MLP: {best_mlp_name}")
    best_simple, best_logit = compare_nn_gbdt_ensemble(oof_cat, best_mlp_oof, target, best_mlp_name)

    # Generate submissions
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSIONS")
    print("=" * 60)

    submission = pd.read_csv('../../sample_submission.csv')

    # CatBoost only
    submission['Drafted'] = pred_cat
    submission.to_csv('submission_cat_only.csv', index=False)
    print("1. submission_cat_only.csv")

    # Best MLP only
    submission['Drafted'] = best_mlp_pred
    submission.to_csv('submission_mlp_only.csv', index=False)
    print("2. submission_mlp_only.csv")

    # Simple average (best weight)
    w_cat = best_simple['w']
    simple_pred = w_cat * pred_cat + (1 - w_cat) * best_mlp_pred
    submission['Drafted'] = simple_pred
    submission.to_csv('submission_simple_avg.csv', index=False)
    print(f"3. submission_simple_avg.csv (CAT={w_cat:.2f})")

    # Logit average (best weight)
    w_cat = best_logit['w']
    logit_pred = logit_average([pred_cat, best_mlp_pred], [w_cat, 1 - w_cat])
    submission['Drafted'] = logit_pred
    submission.to_csv('submission_logit_avg.csv', index=False)
    print(f"4. submission_logit_avg.csv (CAT={w_cat:.2f})")

    print("\nDone!")
