"""
Exp44: Strong Ensemble
- Base: exp33 feature engineering (31 features)
- Models: CatBoost (replication), LightGBM, XGBoost
- Goal: Create a strong ensemble by forcing diverse models to use the same high-quality features.
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 2023, 101, 555, 999]

EXP_DIR = 'experiments/exp44_strong_ensemble'
BASE_DIR = './'

os.makedirs(EXP_DIR, exist_ok=True)

print("=" * 60)
print("Exp44: Strong Ensemble (CatBoost + LightGBM + XGBoost)")
print("=" * 60)

# ==========================================
# Feature Loading (Meta-data)
# ==========================================
# We manually define the best features from exp33 to avoid dependency on previous result files
top30_features = [
    "School_TE",
    "Age_Year_Diff",
    "Broad_Jump_Type_Z",
    "School_Count",
    "Age_x_Momentum",
    "Momentum_Pos_Diff",
    "School",
    "Age_x_Speed",
    "Agility_3cone_Pos_Z",
    "Weight_Type_Z",
    "Bench_Press_Reps_Pos_Z",
    "Weight_Pos_Z",
    "Speed_Score_Year_Rank",
    "BMI_x_Speed",
    "Momentum_Type_Z",
    "BMI_Pos_Diff",
    "Age_div_Explosion",
    "Bench_Press_Reps_Pos_Diff",
    "Position_TE",
    "Work_Rate_Vertical_Type_Z",
    "Age",
    "Year",
    "Height_Pos_Diff",
    "Height_x_Weight",
    "BMI_Type_Z",
    "Agility_3cone_Type_Z",
    "Bench_per_Weight",
    "Sprint_Efficiency",
    "Broad_Jump_Year_Rank",
    "Position_Type_TE"
]
features = top30_features + ['Agility_3cone_Pos_Diff']

print(f"Features: {len(features)}")

# ==========================================
# Target Encoding (Copied from exp33)
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
# Feature Engineering (Copied from exp33)
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
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ==========================================
# Model Parameters
# ==========================================
# Optimized params from exp07_final
cat_params = {
    'learning_rate': 0.07718772443488796,
    'depth': 3,
    'l2_leaf_reg': 0.0033458292447738312,
    'subsample': 0.8523245279212943,
    'min_data_in_leaf': 71,
    'random_strength': 1.2032200146196355,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'iterations': 10000,
    'task_type': 'CPU'
}

lgb_params = {
    'learning_rate': 0.09276457122109245,
    'num_leaves': 111,
    'max_depth': 3,
    'min_child_samples': 41,
    'subsample': 0.4126775090838999,
    'colsample_bytree': 0.4910286943468972,
    'reg_alpha': 0.5363517544276609,
    'reg_lambda': 9.847873834942789,
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'n_estimators': 10000,
    'seed': 42
}

xgb_params = {
    'learning_rate': 0.010557654198243605,
    'max_depth': 5,
    'min_child_weight': 2,
    'subsample': 0.42823789878654434,
    'colsample_bytree': 0.6400167112588607,
    'gamma': 0.7820326628699041,
    'alpha': 0.017218006622693675,
    'lambda': 0.0010527166617541805,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'early_stopping_rounds': 100,
    'seed': 42,
    'n_estimators': 10000,
    'n_jobs': -1
}

# ==========================================
# Training Function
# ==========================================
def train_model(model_name, X, y, X_test, params, cat_features=None):
    print(f"\n--- Training {model_name} ---")
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    
    for seed_idx, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({seed_idx+1}/{len(SEEDS)})")
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            # Setup specific model
            if model_name == 'CatBoost':
                p = params.copy()
                p['random_seed'] = seed
                model = CatBoostClassifier(**p)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], cat_features=cat_features, early_stopping_rounds=100, verbose=False)
                val_pred = model.predict_proba(X_va)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]
                
            elif model_name == 'LightGBM':
                p = params.copy()
                p['seed'] = seed
                # Treat categorical as numeric (already LabelEncoded)
                model = lgb.LGBMClassifier(**p)
                callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], 
                          eval_names=['valid'], callbacks=callbacks)
                
                val_pred = model.predict_proba(X_va)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]
                
            elif model_name == 'XGBoost':
                p = params.copy()
                p['seed'] = seed
                # Treat categorical as numeric (already LabelEncoded)
                p['early_stopping_rounds'] = 100
                model = xgb.XGBClassifier(**p)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                
                val_pred = model.predict_proba(X_va)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]

            oof[va_idx] += val_pred / N_SEEDS
            pred += test_pred / (N_FOLDS * N_SEEDS)
            
    cv_score = roc_auc_score(y, oof)
    print(f"  {model_name} CV: {cv_score:.5f}")
    return oof, pred, cv_score

# ==========================================
# Run Training
# ==========================================
oof_cat, pred_cat, cv_cat = train_model('CatBoost', X_train, y_train, X_test, cat_params, cat_indices)
oof_lgb, pred_lgb, cv_lgb = train_model('LightGBM', X_train, y_train, X_test, lgb_params, cat_indices)
oof_xgb, pred_xgb, cv_xgb = train_model('XGBoost', X_train, y_train, X_test, xgb_params, cat_indices)

# ==========================================
# Optimization
# ==========================================
print("\n--- Ensemble Optimization ---")
def minimize_func(weights):
    final_oof = (weights[0] * oof_cat + weights[1] * oof_lgb + weights[2] * oof_xgb) / np.sum(weights)
    return -roc_auc_score(y_train, final_oof)

init_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1)] * 3
res = minimize(minimize_func, init_weights, bounds=bounds, method='SLSQP', constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

best_weights = res.x
print(f"Best Weights: Cat={best_weights[0]:.3f}, LGB={best_weights[1]:.3f}, XGB={best_weights[2]:.3f}")

# Final predictions
oof_final = (best_weights[0] * oof_cat + best_weights[1] * oof_lgb + best_weights[2] * oof_xgb)
pred_final = (best_weights[0] * pred_cat + best_weights[1] * pred_lgb + best_weights[2] * pred_xgb)

cv_final = roc_auc_score(y_train, oof_final)
print(f"Ensemble CV: {cv_final:.5f}")
print(f"Exp33 CV: 0.85083")
print(f"Improvement: {cv_final - 0.85083:+.5f}")

# ==========================================
# Save Results
# ==========================================
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': pred_final
})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

results = {
    'model': 'Ensemble (Cat+LGB+XGB)',
    'weights': {
        'cat': float(best_weights[0]),
        'lgb': float(best_weights[1]),
        'xgb': float(best_weights[2])
    },
    'cv_individual': {
        'cat': float(cv_cat),
        'lgb': float(cv_lgb),
        'xgb': float(cv_xgb)
    },
    'cv_ensemble': float(cv_final),
    'improvement': float(cv_final - 0.85083)
}

with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved submission to {EXP_DIR}/submission.csv")
