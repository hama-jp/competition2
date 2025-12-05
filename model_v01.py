import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import optuna
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# 定数
N_FOLDS = 5
N_SEEDS = 10
SEEDS = [42, 2023, 101, 555, 999, 123, 777, 88, 33, 1]
OPTUNA_TRIALS = 50

# ==========================================
# 1. データ準備 (Selected Features Only)
# ==========================================
# Permutation Importanceで勝ち残った50個の精鋭特徴量
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
    'Elite_Count', 'Red_Flag_Count', 'Talent_Diff' # スカウト視点（一芸選手）

def get_data():
    print("Loading and preprocessing...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    sample_sub = pd.read_csv('sample_submission.csv')

    train['is_train'] = 1
    test['is_train'] = 0
    test['Drafted'] = np.nan
    data = pd.concat([train, test], sort=False).reset_index(drop=True)

    # --- Feature Engineering ---
    measure_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    data['Missing_Count'] = data[measure_cols].isnull().sum(axis=1) # 勝ち残り特徴量

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

    # --- 追加: スカウト視点の特徴量 (Elite & Red Flags) ---
    # 評価対象のフィジカル項目
    phys_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']

    # ポジションごとの「上位10%（Elite）」と「下位10%（Red Flag）」の閾値を計算
    # ※ Sprint, Agility, Shuttle は「低いほど良い」、他は「高いほど良い」
    lower_is_better = ['Sprint_40yd', 'Agility_3cone', 'Shuttle']

    # 一時的にデータを保存する辞書
    elite_flags = pd.DataFrame(index=data.index)
    red_flags = pd.DataFrame(index=data.index)

    for col in phys_cols:
        # ポジションごとの分位点を計算
        if col in lower_is_better:
            # 数値が低い方が良い指標
            # 上位10% (0.1 quantile), 下位10% (0.9 quantile)
            q10 = data.groupby('Position')[col].transform(lambda x: x.quantile(0.1))
            q90 = data.groupby('Position')[col].transform(lambda x: x.quantile(0.9))

            elite_flags[f'{col}_Elite'] = (data[col] <= q10).astype(int)
            red_flags[f'{col}_Bad'] = (data[col] >= q90).astype(int)
        else:
            # 数値が高い方が良い指標
            # 上位10% (0.9 quantile), 下位10% (0.1 quantile)
            q90 = data.groupby('Position')[col].transform(lambda x: x.quantile(0.9))
            q10 = data.groupby('Position')[col].transform(lambda x: x.quantile(0.1))

            elite_flags[f'{col}_Elite'] = (data[col] >= q90).astype(int)
            red_flags[f'{col}_Bad'] = (data[col] <= q10).astype(int)

    # 「いくつの項目でエリート級か？」
    data['Elite_Count'] = elite_flags.sum(axis=1)

    # 「いくつの項目で足切りレベルか？」
    data['Red_Flag_Count'] = red_flags.sum(axis=1)

    # 「エリート項目数 - 欠点数」 (純粋なポテンシャル)
    data['Talent_Diff'] = data['Elite_Count'] - data['Red_Flag_Count']


    # Features Filtering
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
# 2. Optuna
# ==========================================
def run_optuna(model_name, X, y, features, cat_indices=None):
    print(f"\n--- Optimizing {model_name} ---")

    def objective(trial):
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        if model_name == 'lgb':
            params = {
                'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                'boosting_type': 'gbdt', 'random_state': 42,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
        elif model_name == 'xgb':
            params = {
                'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
                'random_state': 42, 'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5.0),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                'early_stopping_rounds': 50
            }
        elif model_name == 'cat':
            params = {
                'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_seed': 42,
                'verbose': False, 'allow_writing_files': False, 'iterations': 1000,
                'has_time': False,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
                'random_strength': trial.suggest_float('random_strength', 0, 3.0),
            }

        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            if model_name == 'lgb':
                model = lgb.LGBMClassifier(**params, n_estimators=1000)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
                preds = model.predict_proba(X_va)[:, 1]
            elif model_name == 'xgb':
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                preds = model.predict_proba(X_va)[:, 1]
            elif model_name == 'cat':
                tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
                va_pool = Pool(X_va, y_va, cat_features=cat_indices)
                model = CatBoostClassifier(**params)
                model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=50)
                preds = model.predict_proba(X_va)[:, 1]

            scores.append(roc_auc_score(y_va, preds))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print(f"Best {model_name}: {study.best_value:.5f}")
    return study.best_params

# ==========================================
# 3. Training
# ==========================================
def train_model(model_name, params, X, y, X_test, features, cat_indices=None):
    print(f"\n--- Training {model_name} (10 seeds) ---")
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    if model_name == 'lgb':
        params.update({'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'verbosity': -1})
    elif model_name == 'xgb':
        params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'early_stopping_rounds': 100})
    elif model_name == 'cat':
        params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': False, 'allow_writing_files': False, 'iterations': 10000, 'has_time': False})

    for seed in SEEDS:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        if model_name == 'lgb': params['random_state'] = seed
        elif model_name == 'xgb': params['random_state'] = seed
        elif model_name == 'cat': params['random_seed'] = seed

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx][features], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx][features], y.iloc[val_idx]

            if model_name == 'lgb':
                model = lgb.LGBMClassifier(**params, n_estimators=10000)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
            elif model_name == 'xgb':
                model = xgb.XGBClassifier(**params, n_estimators=10000)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif model_name == 'cat':
                tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
                va_pool = Pool(X_val, y_val, cat_features=cat_indices)
                model = CatBoostClassifier(**params)
                model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)

            p_val = model.predict_proba(X_val)[:, 1]
            p_test = model.predict_proba(X_test[features])[:, 1]

            oof[val_idx] += p_val / len(SEEDS)
            preds += p_test / (N_FOLDS * len(SEEDS))

        print(f"Seed {seed} done.", end="\r")

    print(f"\n{model_name} CV: {roc_auc_score(y, oof):.5f}")
    return oof, preds

# --- Main Pipeline ---
train_df, test_df, target, features, cat_indices = get_data()
print(f"Selected Features: {len(features)}")

bp_lgb = run_optuna('lgb', train_df, target, features)
bp_xgb = run_optuna('xgb', train_df, target, features)
bp_cat = run_optuna('cat', train_df, target, features, cat_indices)

oof_lgb, pred_lgb = train_model('lgb', bp_lgb, train_df, target, test_df, features)
oof_xgb, pred_xgb = train_model('xgb', bp_xgb, train_df, target, test_df, features)
oof_cat, pred_cat = train_model('cat', bp_cat, train_df, target, test_df, features, cat_indices)

# --- Ensemble Weight Optimization (Non-negative constraint) ---
def get_score(weights):
    final_oof = (oof_lgb * weights[0]) + (oof_xgb * weights[1]) + (oof_cat * weights[2])
    return -roc_auc_score(target, final_oof)

# 制約付き最適化 (合計1, 各重み0以上)
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bnds = ((0, 1), (0, 1), (0, 1))
init_w = [0.3, 0.3, 0.4] # 初期値はCatBoost強め

res = minimize(get_score, init_w, method='SLSQP', bounds=bnds, constraints=cons)
best_w = res.x

print(f"\n========================================")
print(f"Weights: LGB={best_w[0]:.3f}, XGB={best_w[1]:.3f}, CAT={best_w[2]:.3f}")
print(f"Final CV AUC: {-res.fun:.5f}")
print(f"========================================")

final_preds = (pred_lgb * best_w[0]) + (pred_xgb * best_w[1]) + (pred_cat * best_w[2])
submission = pd.read_csv('sample_submission.csv')
submission['Drafted'] = final_preds
submission.to_csv('submission.csv', index=False)
print("submission.csv created.")

submission_cat = pd.read_csv('sample_submission.csv')
submission_cat['Drafted'] = pred_cat
submission_cat.to_csv('submission_cat_only.csv', index=False)
print("Option 1: submission_cat_only.csv created. (Best CV)")

# 2. 保険モデル: CatBoost主体 + XGBoost補正
# CatBoost 85% + XGBoost 15%
# LightGBMは今回スコアが低めなので外します。
# 汎化性能を重視して、少しだけXGBの視点を入れます。
w_cat = 0.85
w_xgb = 0.15
ensemble_safe = (pred_cat * w_cat) + (pred_xgb * w_xgb)

submission_safe = pd.read_csv('sample_submission.csv')
submission_safe['Drafted'] = ensemble_safe
submission_safe.to_csv('submission_ensemble_safe.csv', index=False)
print("Option 2: submission_ensemble_safe.csv created. (Safe Blend)")

# ================================================
# 1. 最適化された重みで最終的な予測値を計算
final_oof_preds = (oof_lgb * best_w[0]) + (oof_xgb * best_w[1]) + (oof_cat * best_w[2])
final_test_preds = (pred_lgb * best_w[0]) + (pred_xgb * best_w[1]) + (pred_cat * best_w[2])

# 2. 統計量の確認
print("--- Prediction Statistics ---")
print(f"Train (OOF) Mean: {np.mean(final_oof_preds):.4f} | Std: {np.std(final_oof_preds):.4f}")
print(f"Test (Pred) Mean: {np.mean(final_test_preds):.4f} | Std: {np.std(final_test_preds):.4f}")
print("-" * 30)
print(f"Train Positive Ratio (Actual): {target.mean():.4f}")

# 3. 分布の可視化
plt.figure(figsize=(14, 6))

# 左側：TrainとTestの全体分布比較
plt.subplot(1, 2, 1)
sns.kdeplot(final_oof_preds, label='Train OOF (Predictions)', fill=True, color='blue', alpha=0.3)
sns.kdeplot(final_test_preds, label='Test (Submission)', fill=True, color='orange', alpha=0.3)
plt.title('Distribution Comparison: Train OOF vs Test')
plt.xlabel('Predicted Probability (Drafted)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# 右側：Trainデータの正解ラベルごとの予測分布
plt.subplot(1, 2, 2)
sns.kdeplot(final_oof_preds[target == 0], label='Actual: Not Drafted (0)', fill=True, color='red', alpha=0.3)
sns.kdeplot(final_oof_preds[target == 1], label='Actual: Drafted (1)', fill=True, color='green', alpha=0.3)
plt.title('Train OOF Separation (Discriminative Power)')
plt.xlabel('Predicted Probability (Drafted)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
