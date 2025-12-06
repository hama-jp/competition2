"""
Exp11: Pseudo-labeling + Missing Pattern Features
Based on analysis of surprising predictions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Exp11: Pseudo-labeling + Missing Pattern Features")
print("=" * 60)

# Load data
train = pd.read_csv('/home/user/competition2/train.csv')
test = pd.read_csv('/home/user/competition2/test.csv')

# Load exp07 test predictions for pseudo-labeling
pred_exp07 = np.load('/home/user/competition2/experiments/exp07_final/pred_final.npy')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

def create_features(df, is_train=True):
    """Create features with missing pattern analysis"""
    df = df.copy()

    # Physical features
    numeric_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                    'Broad_Jump', 'Agility_3cone', 'Shuttle']

    # === Missing pattern features (key insight from analysis) ===
    df['missing_count'] = df[numeric_cols].isna().sum(axis=1)
    df['missing_ratio'] = df['missing_count'] / len(numeric_cols)

    # Age missing is very important for FN cases
    df['age_missing'] = df['Age'].isna().astype(int)

    # Specific missing patterns
    df['bench_missing'] = df['Bench_Press_Reps'].isna().astype(int)
    df['speed_missing'] = df['Sprint_40yd'].isna().astype(int)
    df['agility_missing'] = (df['Agility_3cone'].isna() | df['Shuttle'].isna()).astype(int)

    # Combine age_missing with other factors
    df['age_missing_but_bench'] = (df['age_missing'] == 1) & (df['Bench_Press_Reps'].notna())
    df['age_missing_but_bench'] = df['age_missing_but_bench'].astype(int)

    # === Physical features ===
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Height_Weight_Ratio'] = df['Height'] / df['Weight']

    # Explosive power indicators
    df['Power_Index'] = df['Vertical_Jump'] * df['Broad_Jump'] / 10000
    df['Speed_Power'] = df['Vertical_Jump'] / (df['Sprint_40yd'] + 0.1)

    # Agility composite
    df['Agility_Composite'] = (df['Shuttle'] + df['Agility_3cone']) / 2

    # === Position-based Z-scores ===
    for col in numeric_cols:
        if col in df.columns:
            pos_mean = df.groupby('Position')[col].transform('mean')
            pos_std = df.groupby('Position')[col].transform('std')
            df[f'{col}_pos_zscore'] = (df[col] - pos_mean) / (pos_std + 1e-8)

    # === School features ===
    # Calculate school draft rate from training data only
    if is_train:
        school_stats = df.groupby('School').agg({
            'Drafted': ['mean', 'count']
        }).reset_index()
        school_stats.columns = ['School', 'school_draft_rate', 'school_count']
        # Store for later use
        school_stats.to_csv('/home/user/competition2/experiments/exp11_pseudo_label/school_stats.csv', index=False)
    else:
        school_stats = pd.read_csv('/home/user/competition2/experiments/exp11_pseudo_label/school_stats.csv')

    df = df.merge(school_stats, on='School', how='left')
    df['school_draft_rate'] = df['school_draft_rate'].fillna(0.5)
    df['school_count'] = df['school_count'].fillna(1)

    # Strong school indicator
    df['strong_school'] = (df['school_draft_rate'] >= 0.65).astype(int)

    # === Interaction features ===
    # Age missing + Strong school = likely drafted
    df['age_missing_strong_school'] = df['age_missing'] * df['strong_school']

    # === Year features ===
    df['Year_normalized'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min() + 1)

    # === Position Type encoding ===
    le_pos_type = LabelEncoder()
    df['Position_Type_encoded'] = le_pos_type.fit_transform(df['Position_Type'].fillna('unknown'))

    le_player_type = LabelEncoder()
    df['Player_Type_encoded'] = le_player_type.fit_transform(df['Player_Type'].fillna('unknown'))

    return df

# Create features
train_fe = create_features(train, is_train=True)
test_fe = create_features(test, is_train=False)

# Define features
feature_cols = [
    # Basic physical
    'Height', 'Weight', 'Age', 'BMI', 'Height_Weight_Ratio',
    # Performance metrics
    'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump',
    'Agility_3cone', 'Shuttle',
    # Composite metrics
    'Power_Index', 'Speed_Power', 'Agility_Composite',
    # Position Z-scores
    'Sprint_40yd_pos_zscore', 'Vertical_Jump_pos_zscore',
    'Bench_Press_Reps_pos_zscore', 'Broad_Jump_pos_zscore',
    'Agility_3cone_pos_zscore', 'Shuttle_pos_zscore',
    # Missing pattern features (NEW)
    'missing_count', 'missing_ratio', 'age_missing',
    'bench_missing', 'speed_missing', 'agility_missing',
    'age_missing_but_bench',
    # School features
    'school_draft_rate', 'school_count', 'strong_school',
    'age_missing_strong_school',
    # Categorical
    'Position_Type_encoded', 'Player_Type_encoded',
    'Year', 'Year_normalized'
]

# Filter to existing columns
features = [col for col in feature_cols if col in train_fe.columns]
print(f"\nUsing {len(features)} features")

# === Pseudo-labeling ===
print("\n" + "=" * 60)
print("Pseudo-labeling setup")
print("=" * 60)

# High confidence predictions from exp07
test_fe['pseudo_label'] = pred_exp07
test_fe['pseudo_confidence'] = np.abs(pred_exp07 - 0.5)

# Select high confidence samples (threshold: 0.85 or 0.15)
high_conf_threshold = 0.35  # confidence > 0.35 means prob > 0.85 or prob < 0.15
pseudo_mask = test_fe['pseudo_confidence'] >= high_conf_threshold
pseudo_data = test_fe[pseudo_mask].copy()
pseudo_data['Drafted'] = (pseudo_data['pseudo_label'] >= 0.5).astype(int)

print(f"High confidence test samples: {len(pseudo_data)} / {len(test_fe)}")
print(f"  Pseudo positive: {(pseudo_data['Drafted'] == 1).sum()}")
print(f"  Pseudo negative: {(pseudo_data['Drafted'] == 0).sum()}")

# Create augmented training set
train_aug = pd.concat([train_fe, pseudo_data], axis=0, ignore_index=True)
print(f"\nOriginal train: {len(train_fe)}")
print(f"Augmented train: {len(train_aug)}")

# === Model Training ===
print("\n" + "=" * 60)
print("Model Training")
print("=" * 60)

# Best params from exp07
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.03306,
    'num_leaves': 45,
    'max_depth': 12,
    'min_child_samples': 18,
    'subsample': 0.7098,
    'colsample_bytree': 0.9046,
    'reg_alpha': 0.06575,
    'reg_lambda': 0.01414,
    'n_estimators': 458,
    'random_state': 42
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.02967,
    'max_depth': 7,
    'min_child_weight': 3,
    'subsample': 0.7968,
    'colsample_bytree': 0.9219,
    'reg_alpha': 0.02327,
    'reg_lambda': 0.01049,
    'n_estimators': 328,
    'random_state': 42,
    'verbosity': 0
}

cat_params = {
    'iterations': 500,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': False
}

def train_with_pseudo_labeling(train_orig, train_aug, test_df, features, n_seeds=5):
    """Train with and without pseudo-labeling"""

    X_orig = train_orig[features].values
    y_orig = train_orig['Drafted'].values

    X_aug = train_aug[features].values
    y_aug = train_aug['Drafted'].values

    X_test = test_df[features].values

    results = {}

    # Method 1: Original training (baseline)
    print("\n--- Method 1: Original Training (No Pseudo-labeling) ---")
    oof_orig = np.zeros(len(train_orig))
    pred_orig = np.zeros(len(test_df))

    for seed in range(n_seeds):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_orig, y_orig)):
            X_tr, X_val = X_orig[train_idx], X_orig[val_idx]
            y_tr, y_val = y_orig[train_idx], y_orig[val_idx]

            # LGB
            lgb_params_seed = lgb_params.copy()
            lgb_params_seed['random_state'] = seed * 100 + fold
            model_lgb = lgb.LGBMClassifier(**lgb_params_seed)
            model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

            # XGB
            xgb_params_seed = xgb_params.copy()
            xgb_params_seed['random_state'] = seed * 100 + fold
            model_xgb = xgb.XGBClassifier(**xgb_params_seed)
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            # CatBoost
            cat_params_seed = cat_params.copy()
            cat_params_seed['random_seed'] = seed * 100 + fold
            model_cat = CatBoostClassifier(**cat_params_seed)
            model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)

            # Ensemble
            pred_val = (model_lgb.predict_proba(X_val)[:, 1] +
                       model_xgb.predict_proba(X_val)[:, 1] +
                       model_cat.predict_proba(X_val)[:, 1]) / 3
            pred_test = (model_lgb.predict_proba(X_test)[:, 1] +
                        model_xgb.predict_proba(X_test)[:, 1] +
                        model_cat.predict_proba(X_test)[:, 1]) / 3

            oof_orig[val_idx] += pred_val / n_seeds
            pred_orig += pred_test / (5 * n_seeds)

    cv_orig = roc_auc_score(y_orig, oof_orig)
    print(f"CV (Original): {cv_orig:.5f}")
    results['original'] = {'cv': cv_orig, 'pred': pred_orig}

    # Method 2: With Pseudo-labeling
    print("\n--- Method 2: With Pseudo-labeling ---")
    # For pseudo-labeling, we train on augmented data but validate on original
    oof_pseudo = np.zeros(len(train_orig))
    pred_pseudo = np.zeros(len(test_df))

    for seed in range(n_seeds):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_orig, y_orig)):
            # Training includes original + pseudo-labeled
            X_tr_orig, X_val = X_orig[train_idx], X_orig[val_idx]
            y_tr_orig, y_val = y_orig[train_idx], y_orig[val_idx]

            # Add pseudo-labeled samples to training
            X_pseudo = train_aug[train_aug.index >= len(train_orig)][features].values
            y_pseudo = train_aug[train_aug.index >= len(train_orig)]['Drafted'].values

            X_tr = np.vstack([X_tr_orig, X_pseudo])
            y_tr = np.concatenate([y_tr_orig, y_pseudo])

            # LGB
            lgb_params_seed = lgb_params.copy()
            lgb_params_seed['random_state'] = seed * 100 + fold
            model_lgb = lgb.LGBMClassifier(**lgb_params_seed)
            model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

            # XGB
            xgb_params_seed = xgb_params.copy()
            xgb_params_seed['random_state'] = seed * 100 + fold
            model_xgb = xgb.XGBClassifier(**xgb_params_seed)
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            # CatBoost
            cat_params_seed = cat_params.copy()
            cat_params_seed['random_seed'] = seed * 100 + fold
            model_cat = CatBoostClassifier(**cat_params_seed)
            model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)

            # Ensemble
            pred_val = (model_lgb.predict_proba(X_val)[:, 1] +
                       model_xgb.predict_proba(X_val)[:, 1] +
                       model_cat.predict_proba(X_val)[:, 1]) / 3
            pred_test = (model_lgb.predict_proba(X_test)[:, 1] +
                        model_xgb.predict_proba(X_test)[:, 1] +
                        model_cat.predict_proba(X_test)[:, 1]) / 3

            oof_pseudo[val_idx] += pred_val / n_seeds
            pred_pseudo += pred_test / (5 * n_seeds)

    cv_pseudo = roc_auc_score(y_orig, oof_pseudo)
    print(f"CV (Pseudo-labeling): {cv_pseudo:.5f}")
    results['pseudo'] = {'cv': cv_pseudo, 'pred': pred_pseudo}

    return results

# Train models
results = train_with_pseudo_labeling(train_fe, train_aug, test_fe, features, n_seeds=5)

# === Results ===
print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)
print(f"CV (Original + New Features): {results['original']['cv']:.5f}")
print(f"CV (Pseudo-labeling): {results['pseudo']['cv']:.5f}")
print(f"Exp07 baseline: 0.84651")

# Select best
if results['pseudo']['cv'] > results['original']['cv']:
    best_method = 'pseudo'
    best_cv = results['pseudo']['cv']
    best_pred = results['pseudo']['pred']
else:
    best_method = 'original'
    best_cv = results['original']['cv']
    best_pred = results['original']['pred']

print(f"\nBest: {best_method} (CV={best_cv:.5f})")

# Save results
output_dir = '/home/user/competition2/experiments/exp11_pseudo_label'

# Submission
submission = pd.DataFrame({
    'Id': test['Id'],
    'Drafted': best_pred
})
submission.to_csv(f'{output_dir}/submission.csv', index=False)

# Also save both versions
submission_orig = pd.DataFrame({
    'Id': test['Id'],
    'Drafted': results['original']['pred']
})
submission_orig.to_csv(f'{output_dir}/submission_original.csv', index=False)

submission_pseudo = pd.DataFrame({
    'Id': test['Id'],
    'Drafted': results['pseudo']['pred']
})
submission_pseudo.to_csv(f'{output_dir}/submission_pseudo.csv', index=False)

# Save results
results_summary = {
    'cv_original': results['original']['cv'],
    'cv_pseudo': results['pseudo']['cv'],
    'best_method': best_method,
    'best_cv': best_cv,
    'n_pseudo_samples': len(pseudo_data),
    'features_used': features
}
with open(f'{output_dir}/results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to {output_dir}")
print(f"Final CV: {best_cv:.5f}")
