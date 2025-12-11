"""
Exp42: Logit-Mean Ensemble (ロジット平均アンサンブル)

通常の確率平均:  final = w1*p1 + w2*p2 + w3*p3
ロジット平均:    final = sigmoid(w1*logit(p1) + w2*logit(p2) + w3*logit(p3))

理論:
- 確率を直接平均すると、0.5付近と0/1付近が同じ重みで扱われる
- ロジット空間では、0.5付近は細かく、0/1付近は緩やかに変化
- 異なるキャリブレーションのモデルを混ぜるときに有効

仮説:
- 勾配ブースティング系（LGB/XGB/CAT）だけだと効果薄？
- 同系統モデルは確率キャリブレーションが似ているため
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from scipy.special import logit, expit  # logit = log(p/(1-p)), expit = sigmoid
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5  # 高速化のため5に
SEEDS = [42, 2023, 101, 555, 999]

# ==========================================
# Logit Average Functions
# ==========================================
def safe_logit(p, eps=1e-7):
    """安全なロジット変換（0/1でinfを避ける）"""
    p = np.clip(p, eps, 1 - eps)
    return logit(p)

def logit_average(probs_list, weights):
    """
    ロジット空間での加重平均
    probs_list: list of probability arrays
    weights: array of weights (sum to 1)
    """
    logits = [safe_logit(p) for p in probs_list]
    weighted_logit = sum(w * l for w, l in zip(weights, logits))
    return expit(weighted_logit)

def simple_average(probs_list, weights):
    """通常の確率空間での加重平均"""
    return sum(w * p for w, p in zip(weights, probs_list))

# ==========================================
# Data Preparation (Same as model_v01)
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
# Quick Training (Default Params)
# ==========================================
def train_lgb(X, y, X_test, features):
    print("Training LightGBM...")
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'verbosity': -1, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6
    }
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        params['random_state'] = seed
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx][features], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx][features], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params, n_estimators=5000)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])

            oof[val_idx] += model.predict_proba(X_val)[:, 1] / N_SEEDS
            preds += model.predict_proba(X_test[features])[:, 1] / (N_FOLDS * N_SEEDS)

    print(f"LGB CV: {roc_auc_score(y, oof):.5f}")
    return oof, preds

def train_xgb(X, y, X_test, features):
    print("Training XGBoost...")
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'learning_rate': 0.05, 'max_depth': 6, 'early_stopping_rounds': 100
    }
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for seed in SEEDS:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        params['random_state'] = seed
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx][features], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx][features], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params, n_estimators=5000)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            oof[val_idx] += model.predict_proba(X_val)[:, 1] / N_SEEDS
            preds += model.predict_proba(X_test[features])[:, 1] / (N_FOLDS * N_SEEDS)

    print(f"XGB CV: {roc_auc_score(y, oof):.5f}")
    return oof, preds

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

    print(f"CAT CV: {roc_auc_score(y, oof):.5f}")
    return oof, preds

# ==========================================
# Comparison: Simple vs Logit Average
# ==========================================
def compare_ensemble_methods(oof_lgb, oof_xgb, oof_cat, target):
    """単純平均 vs ロジット平均の比較"""

    print("\n" + "=" * 60)
    print("ENSEMBLE COMPARISON: Simple Average vs Logit Average")
    print("=" * 60)

    results = []

    # --- Grid Search over Weights ---
    print("\n--- Weight Grid Search ---")
    best_simple = {'score': 0, 'weights': None}
    best_logit = {'score': 0, 'weights': None}

    for w1 in np.arange(0, 1.05, 0.1):
        for w2 in np.arange(0, 1.05 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue

            weights = [w1, w2, w3]
            probs_list = [oof_lgb, oof_xgb, oof_cat]

            # Simple average
            simple_pred = simple_average(probs_list, weights)
            simple_score = roc_auc_score(target, simple_pred)

            # Logit average
            logit_pred = logit_average(probs_list, weights)
            logit_score = roc_auc_score(target, logit_pred)

            if simple_score > best_simple['score']:
                best_simple = {'score': simple_score, 'weights': weights}
            if logit_score > best_logit['score']:
                best_logit = {'score': logit_score, 'weights': weights}

    print(f"\nBest Simple Average:")
    print(f"  Weights: LGB={best_simple['weights'][0]:.1f}, XGB={best_simple['weights'][1]:.1f}, CAT={best_simple['weights'][2]:.1f}")
    print(f"  CV AUC: {best_simple['score']:.5f}")

    print(f"\nBest Logit Average:")
    print(f"  Weights: LGB={best_logit['weights'][0]:.1f}, XGB={best_logit['weights'][1]:.1f}, CAT={best_logit['weights'][2]:.1f}")
    print(f"  CV AUC: {best_logit['score']:.5f}")

    diff = best_logit['score'] - best_simple['score']
    print(f"\nDifference (Logit - Simple): {diff:+.5f}")

    # --- Same Weight Comparison ---
    print("\n--- Same Weight Comparison ---")
    test_weights = [
        [1.0, 0.0, 0.0],  # LGB only
        [0.0, 1.0, 0.0],  # XGB only
        [0.0, 0.0, 1.0],  # CAT only
        [0.33, 0.33, 0.34],  # Equal
        [0.2, 0.2, 0.6],  # CAT heavy
        [0.3, 0.3, 0.4],  # Mild CAT heavy
    ]

    print(f"{'Weights':<25} {'Simple':>10} {'Logit':>10} {'Diff':>10}")
    print("-" * 55)

    for weights in test_weights:
        probs_list = [oof_lgb, oof_xgb, oof_cat]
        simple_pred = simple_average(probs_list, weights)
        logit_pred = logit_average(probs_list, weights)

        simple_score = roc_auc_score(target, simple_pred)
        logit_score = roc_auc_score(target, logit_pred)
        diff = logit_score - simple_score

        w_str = f"[{weights[0]:.1f}, {weights[1]:.1f}, {weights[2]:.1f}]"
        print(f"{w_str:<25} {simple_score:>10.5f} {logit_score:>10.5f} {diff:>+10.5f}")

    # --- 分析: 確率分布の違い ---
    print("\n--- Probability Distribution Analysis ---")
    print(f"LGB:  mean={np.mean(oof_lgb):.4f}, std={np.std(oof_lgb):.4f}, min={np.min(oof_lgb):.4f}, max={np.max(oof_lgb):.4f}")
    print(f"XGB:  mean={np.mean(oof_xgb):.4f}, std={np.std(oof_xgb):.4f}, min={np.min(oof_xgb):.4f}, max={np.max(oof_xgb):.4f}")
    print(f"CAT:  mean={np.mean(oof_cat):.4f}, std={np.std(oof_cat):.4f}, min={np.min(oof_cat):.4f}, max={np.max(oof_cat):.4f}")

    # 極端な予測の割合
    print("\n--- Extreme Prediction Ratio ---")
    for name, oof in [('LGB', oof_lgb), ('XGB', oof_xgb), ('CAT', oof_cat)]:
        extreme_low = np.mean(oof < 0.1) * 100
        extreme_high = np.mean(oof > 0.9) * 100
        print(f"{name}: <0.1 = {extreme_low:.1f}%, >0.9 = {extreme_high:.1f}%")

    return best_simple, best_logit

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    train_df, test_df, target, features, cat_indices = get_data()
    print(f"Features: {len(features)}")

    # Train all models
    oof_lgb, pred_lgb = train_lgb(train_df, target, test_df, features)
    oof_xgb, pred_xgb = train_xgb(train_df, target, test_df, features)
    oof_cat, pred_cat = train_cat(train_df, target, test_df, features, cat_indices)

    # Compare ensemble methods
    best_simple, best_logit = compare_ensemble_methods(oof_lgb, oof_xgb, oof_cat, target)

    # Generate submissions
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSIONS")
    print("=" * 60)

    submission = pd.read_csv('../../sample_submission.csv')

    # Simple average with best weights
    simple_final = simple_average([pred_lgb, pred_xgb, pred_cat], best_simple['weights'])
    submission['Drafted'] = simple_final
    submission.to_csv('submission_simple_avg.csv', index=False)
    print(f"1. submission_simple_avg.csv (weights={best_simple['weights']})")

    # Logit average with best weights
    logit_final = logit_average([pred_lgb, pred_xgb, pred_cat], best_logit['weights'])
    submission['Drafted'] = logit_final
    submission.to_csv('submission_logit_avg.csv', index=False)
    print(f"2. submission_logit_avg.csv (weights={best_logit['weights']})")

    # CatBoost only (baseline)
    submission['Drafted'] = pred_cat
    submission.to_csv('submission_cat_only.csv', index=False)
    print("3. submission_cat_only.csv (CatBoost only baseline)")

    print("\nDone!")
