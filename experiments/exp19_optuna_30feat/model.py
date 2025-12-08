"""
Exp19: Optuna Hyperparameter Tuning for 30 Features
- Re-tune hyperparameters specifically for the 30 selected features
- Focus on regularization to reduce CV-LB gap
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 5
OPTUNA_TRIALS = 50

BASE_DIR = '/home/user/competition2'
EXP_DIR = '/home/user/competition2/experiments/exp19_optuna_30feat'

print("=" * 60)
print("Exp19: Optuna Tuning for 30 Features")
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
# Optuna Optimization
# ==========================================
def optimize_lgb(X, y, features):
    print("Optimizing LightGBM...")

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            model = lgb.LGBMClassifier(**params, n_estimators=2000)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    print(f"  Best LGB CV: {study.best_value:.5f}")
    return study.best_params

def optimize_xgb(X, y, features):
    print("Optimizing XGBoost...")

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'gamma': trial.suggest_float('gamma', 0, 5.0),
            'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
            'lambda': trial.suggest_float('lambda', 0.1, 10.0, log=True),
        }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            model = xgb.XGBClassifier(**params, n_estimators=2000, early_stopping_rounds=50)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    print(f"  Best XGB CV: {study.best_value:.5f}")
    return study.best_params

def optimize_cat(X, y, features, cat_indices):
    print("Optimizing CatBoost...")

    def objective(trial):
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False,
            'allow_writing_files': False,
            'random_seed': 42,
            'iterations': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'random_strength': trial.suggest_float('random_strength', 0, 3.0),
        }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)

            model = CatBoostClassifier(**params)
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=50)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    print(f"  Best CAT CV: {study.best_value:.5f}")
    return study.best_params

# ==========================================
# Training with optimized params
# ==========================================
def train_final(X, y, X_test, features, cat_indices, lgb_params, xgb_params, cat_params, n_seeds=5):
    seeds = [42, 2023, 101, 555, 999]

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    pred_lgb = np.zeros(len(X_test))
    pred_xgb = np.zeros(len(X_test))
    pred_cat = np.zeros(len(X_test))

    print("\nTraining final models with 5 seeds...")
    for seed in seeds:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx][features], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx][features], y.iloc[va_idx]

            # LGB
            lgb_p = lgb_params.copy()
            lgb_p.update({'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'verbosity': -1})
            lgb_p['random_state'] = seed
            model_lgb = lgb.LGBMClassifier(**lgb_p, n_estimators=10000)
            model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_lgb[va_idx] += model_lgb.predict_proba(X_va)[:, 1] / n_seeds
            pred_lgb += model_lgb.predict_proba(X_test[features])[:, 1] / (5 * n_seeds)

            # XGB
            xgb_p = xgb_params.copy()
            xgb_p.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'early_stopping_rounds': 100})
            xgb_p['random_state'] = seed
            model_xgb = xgb.XGBClassifier(**xgb_p, n_estimators=10000)
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_xgb[va_idx] += model_xgb.predict_proba(X_va)[:, 1] / n_seeds
            pred_xgb += model_xgb.predict_proba(X_test[features])[:, 1] / (5 * n_seeds)

            # CatBoost
            cat_p = cat_params.copy()
            cat_p.update({'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': False, 'allow_writing_files': False, 'iterations': 10000})
            cat_p['random_seed'] = seed
            model_cat = CatBoostClassifier(**cat_p)
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            va_pool = Pool(X_va, y_va, cat_features=cat_indices)
            model_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=100)
            oof_cat[va_idx] += model_cat.predict_proba(X_va)[:, 1] / n_seeds
            pred_cat += model_cat.predict_proba(X_test[features])[:, 1] / (5 * n_seeds)

    return oof_lgb, oof_xgb, oof_cat, pred_lgb, pred_xgb, pred_cat

# ==========================================
# Main
# ==========================================
print("Loading data...")
train_df, test_df, target = get_data()

# Load top 30 features
with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
    exp13_results = json.load(f)
top_30_features = exp13_results['best_features']

features = [f for f in top_30_features if f in train_df.columns]
cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']
cat_indices = [features.index(c) for c in cat_cols if c in features]

print(f"Features: {len(features)}")

# Optimize hyperparameters
print("\n" + "=" * 60)
print("Optuna Optimization (50 trials each)")
print("=" * 60)

best_lgb_params = optimize_lgb(train_df, target, features)
best_xgb_params = optimize_xgb(train_df, target, features)
best_cat_params = optimize_cat(train_df, target, features, cat_indices)

# Train final models
print("\n" + "=" * 60)
print("Training Final Models")
print("=" * 60)

oof_lgb, oof_xgb, oof_cat, pred_lgb, pred_xgb, pred_cat = train_final(
    train_df, target, test_df, features, cat_indices,
    best_lgb_params, best_xgb_params, best_cat_params
)

cv_lgb = roc_auc_score(target, oof_lgb)
cv_xgb = roc_auc_score(target, oof_xgb)
cv_cat = roc_auc_score(target, oof_cat)

print(f"\nIndividual CVs:")
print(f"  LGB: {cv_lgb:.5f}")
print(f"  XGB: {cv_xgb:.5f}")
print(f"  CAT: {cv_cat:.5f}")

# Ensemble
oof_final = (oof_lgb + oof_xgb + oof_cat) / 3
pred_final = (pred_lgb + pred_xgb + pred_cat) / 3
cv_final = roc_auc_score(target, oof_final)

print(f"  Ensemble: {cv_final:.5f}")
print(f"\nExp13 reference: CV = 0.85138, LB = 0.84524")

# Save
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': pred_final
})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

# Save results
results = {
    'cv_lgb': cv_lgb,
    'cv_xgb': cv_xgb,
    'cv_cat': cv_cat,
    'cv_final': cv_final,
    'best_lgb_params': best_lgb_params,
    'best_xgb_params': best_xgb_params,
    'best_cat_params': best_cat_params,
    'exp13_cv': 0.85138,
    'exp13_lb': 0.84524
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {EXP_DIR}")
