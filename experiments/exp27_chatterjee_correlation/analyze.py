"""
Exp27: Chatterjee Correlation Analysis for Feature Selection
- Implement Chatterjee's xi (ξ) correlation coefficient
- Compare with Pearson and Spearman correlations
- Identify features with non-linear relationships to target
- Propose feature selection improvements
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
import warnings
warnings.filterwarnings('ignore')

EXP_DIR = '/home/user/competition2/experiments/exp27_chatterjee_correlation'
BASE_DIR = '/home/user/competition2'

print("=" * 60)
print("Exp27: Chatterjee Correlation Analysis")
print("=" * 60)

# ==========================================
# Chatterjee Correlation Implementation
# ==========================================
def chatterjee_correlation(x, y):
    """
    Calculate Chatterjee's xi (ξ) correlation coefficient.

    ξ(X, Y) measures whether Y is a function of X (not symmetric).
    - ξ = 1 when Y is a measurable function of X
    - ξ → 0 when X and Y are independent (for large n)

    Reference: Chatterjee (2021) "A New Coefficient of Correlation"

    Parameters:
    -----------
    x, y : array-like
        Input arrays (must be same length)

    Returns:
    --------
    xi : float
        Chatterjee correlation coefficient (0 to 1)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 3:
        return np.nan

    # Sort by x, get corresponding y ranks
    order = np.argsort(x)
    y_sorted = y[order]

    # Get ranks of y (handling ties by average rank)
    r = stats.rankdata(y_sorted, method='average')

    # Calculate sum of absolute differences of consecutive ranks
    l = np.abs(np.diff(r))

    # Calculate the denominator (for ties)
    # Using the formula: 2 * sum(l_i * (n - l_i)) where l_i = rank of y_i
    # Simplified for no/few ties: (n^2 - 1) / 3

    # Standard formula without ties
    xi = 1 - (3 * np.sum(l)) / (n**2 - 1)

    return xi


def chatterjee_correlation_with_ties(x, y):
    """
    Chatterjee correlation with proper tie handling.

    Uses the formula from the original paper that handles ties in Y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 3:
        return np.nan

    # Sort by x
    order = np.argsort(x)
    y_sorted = y[order]

    # Get ranks of y
    r = stats.rankdata(y_sorted, method='max')  # Use max for ties

    # Calculate l_i: number of j such that y_j <= y_i
    l = stats.rankdata(y_sorted, method='max')

    # Numerator: sum of |r_{i+1} - r_i|
    numerator = n * np.sum(np.abs(np.diff(r)))

    # Denominator: 2 * sum(l_i * (n - l_i))
    denominator = 2 * np.sum(l * (n - l))

    if denominator == 0:
        return 0.0

    xi = 1 - numerator / denominator

    return xi


# ==========================================
# Target Encoding (from exp07/exp13)
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
# Full Feature Engineering (from exp07/exp13)
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
    cat_indices = [features.index(c) for c in cat_cols if c in features]

    return train_df, test_df, target, features, cat_cols


# ==========================================
# Load data
# ==========================================
print("\nLoading and preprocessing data...")
train_df, test_df, target, all_features, cat_cols = get_data()
print(f"Total features: {len(all_features)}")
print(f"Training samples: {len(train_df)}")

# ==========================================
# Calculate all correlation types
# ==========================================
print("\n" + "=" * 60)
print("Computing Correlations with Target (Drafted)")
print("=" * 60)

numeric_features = [f for f in all_features if train_df[f].dtype in ['float64', 'int64'] and f not in cat_cols]
print(f"Numeric features: {len(numeric_features)}")

correlation_results = []
y = target.values

print("\nCalculating correlations...")
for i, feat in enumerate(numeric_features):
    x = train_df[feat].values

    # Skip if too many NaN
    valid_mask = ~np.isnan(x)
    if valid_mask.sum() < 100:
        continue

    # Pearson correlation (linear)
    pearson_r, pearson_p = stats.pearsonr(x[valid_mask], y[valid_mask])

    # Spearman correlation (monotonic)
    spearman_r, spearman_p = stats.spearmanr(x[valid_mask], y[valid_mask])

    # Chatterjee correlation (functional relationship)
    chatterjee_xi = chatterjee_correlation(x, y)
    chatterjee_xi_ties = chatterjee_correlation_with_ties(x, y)

    correlation_results.append({
        'feature': feat,
        'pearson': abs(pearson_r),
        'pearson_signed': pearson_r,
        'spearman': abs(spearman_r),
        'spearman_signed': spearman_r,
        'chatterjee': chatterjee_xi,
        'chatterjee_ties': chatterjee_xi_ties,
        'n_valid': valid_mask.sum()
    })

    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{len(numeric_features)} features...")

df_corr = pd.DataFrame(correlation_results)
print(f"\nProcessed {len(df_corr)} numeric features")

# ==========================================
# Analysis: Compare correlation methods
# ==========================================
print("\n" + "=" * 60)
print("Top 30 Features by Each Correlation Method")
print("=" * 60)

# Top by Pearson
print("\n--- Top 20 by Pearson (|r|) ---")
top_pearson = df_corr.nlargest(20, 'pearson')[['feature', 'pearson', 'spearman', 'chatterjee']]
print(top_pearson.to_string(index=False))

# Top by Spearman
print("\n--- Top 20 by Spearman (|rho|) ---")
top_spearman = df_corr.nlargest(20, 'spearman')[['feature', 'pearson', 'spearman', 'chatterjee']]
print(top_spearman.to_string(index=False))

# Top by Chatterjee
print("\n--- Top 20 by Chatterjee (xi) ---")
top_chatterjee = df_corr.nlargest(20, 'chatterjee')[['feature', 'pearson', 'spearman', 'chatterjee']]
print(top_chatterjee.to_string(index=False))

# ==========================================
# Find features with non-linear relationships
# ==========================================
print("\n" + "=" * 60)
print("Features with Potential Non-linear Relationships")
print("=" * 60)
print("(High Chatterjee, Low Pearson = Non-linear relationship)")

df_corr['nonlinear_score'] = df_corr['chatterjee'] - df_corr['pearson']
df_corr['chatterjee_vs_spearman'] = df_corr['chatterjee'] - df_corr['spearman']

# Features where Chatterjee > Pearson significantly
nonlinear_features = df_corr[df_corr['nonlinear_score'] > 0.02].nlargest(20, 'nonlinear_score')
print("\n--- Features with Chatterjee >> Pearson ---")
print(f"{'Feature':40} {'Chatterjee':>10} {'Pearson':>10} {'Diff':>10}")
print("-" * 75)
for _, row in nonlinear_features.iterrows():
    print(f"{row['feature']:40} {row['chatterjee']:10.4f} {row['pearson']:10.4f} {row['nonlinear_score']:10.4f}")

# ==========================================
# Load exp13 results for comparison
# ==========================================
print("\n" + "=" * 60)
print("Comparison with Feature Importance (exp13)")
print("=" * 60)

try:
    with open('/home/user/competition2/experiments/exp13_feature_selection/results.json', 'r') as f:
        exp13_results = json.load(f)
    top30_features = exp13_results['best_features']
    print(f"Current Top 30 features from exp13: {len(top30_features)}")

    # Check which top30 features have high/low Chatterjee
    top30_chatterjee = df_corr[df_corr['feature'].isin(top30_features)]

    print("\n--- Top30 features ranked by Chatterjee ---")
    top30_sorted = top30_chatterjee.nlargest(30, 'chatterjee')
    print(f"{'Feature':40} {'Chatterjee':>10} {'Pearson':>10} {'Spearman':>10}")
    print("-" * 80)
    for _, row in top30_sorted.iterrows():
        print(f"{row['feature']:40} {row['chatterjee']:10.4f} {row['pearson']:10.4f} {row['spearman']:10.4f}")

    # Features NOT in top30 but with high Chatterjee
    not_in_top30 = df_corr[~df_corr['feature'].isin(top30_features)]
    high_chatterjee_candidates = not_in_top30.nlargest(20, 'chatterjee')

    print("\n--- High Chatterjee features NOT in Top30 (Potential additions) ---")
    print(f"{'Feature':40} {'Chatterjee':>10} {'Pearson':>10} {'Spearman':>10}")
    print("-" * 80)
    for _, row in high_chatterjee_candidates.iterrows():
        print(f"{row['feature']:40} {row['chatterjee']:10.4f} {row['pearson']:10.4f} {row['spearman']:10.4f}")

except FileNotFoundError:
    print("exp13 results not found, skipping comparison")
    top30_features = []

# ==========================================
# Composite Score for Feature Selection
# ==========================================
print("\n" + "=" * 60)
print("Composite Score Ranking")
print("=" * 60)
print("Score = 0.4*Chatterjee + 0.3*Spearman + 0.3*Pearson")

df_corr['composite_score'] = (
    0.4 * df_corr['chatterjee'] +
    0.3 * df_corr['spearman'] +
    0.3 * df_corr['pearson']
)

top_composite = df_corr.nlargest(30, 'composite_score')
print(f"\n{'Rank':>4} {'Feature':40} {'Composite':>10} {'Chatterjee':>10} {'Spearman':>10} {'Pearson':>10}")
print("-" * 110)
for i, (_, row) in enumerate(top_composite.iterrows(), 1):
    in_top30 = '*' if row['feature'] in top30_features else ' '
    print(f"{i:4}{in_top30} {row['feature']:40} {row['composite_score']:10.4f} {row['chatterjee']:10.4f} {row['spearman']:10.4f} {row['pearson']:10.4f}")

# ==========================================
# Feature-Feature Chatterjee Matrix (for redundancy detection)
# ==========================================
print("\n" + "=" * 60)
print("Feature-Feature Chatterjee (Top features only)")
print("=" * 60)
print("Detecting non-linear redundancy between features...")

top_features = df_corr.nlargest(15, 'composite_score')['feature'].tolist()

print(f"\nComputing pairwise Chatterjee for top 15 features...")
feature_feature_chatterjee = {}
for i, f1 in enumerate(top_features):
    for j, f2 in enumerate(top_features):
        if i < j:
            x1 = train_df[f1].values
            x2 = train_df[f2].values
            xi_12 = chatterjee_correlation(x1, x2)
            xi_21 = chatterjee_correlation(x2, x1)
            feature_feature_chatterjee[(f1, f2)] = (xi_12, xi_21)

print("\n--- High Chatterjee pairs (potential redundancy) ---")
print(f"{'Feature 1':35} {'Feature 2':35} {'xi(1->2)':>10} {'xi(2->1)':>10}")
print("-" * 95)
sorted_pairs = sorted(feature_feature_chatterjee.items(), key=lambda x: max(x[1]), reverse=True)
for (f1, f2), (xi_12, xi_21) in sorted_pairs[:15]:
    print(f"{f1:35} {f2:35} {xi_12:10.4f} {xi_21:10.4f}")

# ==========================================
# Save results
# ==========================================
results = {
    'correlation_stats': df_corr.to_dict(orient='records'),
    'top30_by_chatterjee': df_corr.nlargest(30, 'chatterjee')['feature'].tolist(),
    'top30_by_composite': df_corr.nlargest(30, 'composite_score')['feature'].tolist(),
    'nonlinear_candidates': df_corr[df_corr['nonlinear_score'] > 0.02]['feature'].tolist(),
    'feature_feature_chatterjee': {f"{k[0]}|{k[1]}": list(v) for k, v in feature_feature_chatterjee.items()}
}

with open(f'{EXP_DIR}/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save correlation dataframe as CSV
df_corr.to_csv(f'{EXP_DIR}/correlation_comparison.csv', index=False)

print(f"\n" + "=" * 60)
print("Results saved to:")
print(f"  {EXP_DIR}/analysis_results.json")
print(f"  {EXP_DIR}/correlation_comparison.csv")
print("=" * 60)

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Chatterjee correlation (xi) can capture non-linear relationships that
Pearson and Spearman miss. Key findings:

1. Features with high Chatterjee correlation may have non-linear
   predictive power for the target variable.

2. The asymmetry of Chatterjee (xi(X,Y) != xi(Y,X)) is useful for
   understanding directional relationships.

3. For feature selection, consider:
   - Adding features with high Chatterjee but not in current Top30
   - Removing features with low Chatterjee (weak functional relationship)
   - Using Chatterjee for non-linear redundancy detection

Recommended next steps:
1. Validate high-Chatterjee features with CV experiments
2. Consider feature transformations for non-linear relationships
3. Test Chatterjee-based feature selection vs importance-based
""")
