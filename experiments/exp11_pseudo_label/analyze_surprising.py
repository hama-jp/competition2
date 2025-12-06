"""
Analyze surprising predictions - players where model was confidently wrong
"""
import pandas as pd
import numpy as np
import json

# Load data
train = pd.read_csv('/home/user/competition2/train.csv')

# Load OOF predictions from exp07 (best GBDT model)
oof_lgb = np.load('/home/user/competition2/experiments/exp07_final/oof_lgb.npy')
oof_xgb = np.load('/home/user/competition2/experiments/exp07_final/oof_xgb.npy')
oof_cat = np.load('/home/user/competition2/experiments/exp07_final/oof_cat.npy')

# Use ensemble OOF (same weights as exp07)
oof_ensemble = (oof_lgb + oof_xgb + oof_cat) / 3

train['oof_pred'] = oof_ensemble
train['actual'] = train['Drafted']
train['error'] = np.abs(train['oof_pred'] - train['actual'])

# Categorize predictions
train['pred_label'] = (train['oof_pred'] >= 0.5).astype(int)
train['correct'] = train['pred_label'] == train['actual']

# Find most surprising cases
# Type 1: Predicted Drafted (high prob) but NOT drafted
false_positives = train[(train['oof_pred'] >= 0.7) & (train['actual'] == 0)].sort_values('oof_pred', ascending=False)

# Type 2: Predicted NOT Drafted (low prob) but WAS drafted
false_negatives = train[(train['oof_pred'] <= 0.3) & (train['actual'] == 1)].sort_values('oof_pred', ascending=True)

print("=" * 70)
print("モデルが意外に思った選手の分析")
print("=" * 70)

# Key features to analyze (using correct column names)
key_features = ['Position', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump',
                'Bench_Press_Reps', 'Broad_Jump', 'Shuttle', 'Agility_3cone', 'School']

print(f"\n【Type 1】ドラフトされると予測したが、されなかった選手 ({len(false_positives)}人)")
print("-" * 70)

if len(false_positives) > 0:
    print(f"\n上位10人の詳細:")
    for i, (idx, row) in enumerate(false_positives.head(10).iterrows()):
        print(f"\n{i+1}. 予測確率: {row['oof_pred']:.3f} (実際: ドラフト外)")
        print(f"   Position: {row['Position']}, School: {row['School']}")
        print(f"   Height: {row['Height']}, Weight: {row['Weight']}, Age: {row['Age']}")
        print(f"   40yd: {row['Sprint_40yd']}, Vertical: {row['Vertical_Jump']}, Bench: {row['Bench_Press_Reps']}")
        print(f"   Broad Jump: {row['Broad_Jump']}, Shuttle: {row['Shuttle']}, 3Cone: {row['Agility_3cone']}")

print(f"\n\n【Type 2】ドラフトされないと予測したが、された選手 ({len(false_negatives)}人)")
print("-" * 70)

if len(false_negatives) > 0:
    print(f"\n上位10人の詳細:")
    for i, (idx, row) in enumerate(false_negatives.head(10).iterrows()):
        print(f"\n{i+1}. 予測確率: {row['oof_pred']:.3f} (実際: ドラフト)")
        print(f"   Position: {row['Position']}, School: {row['School']}")
        print(f"   Height: {row['Height']}, Weight: {row['Weight']}, Age: {row['Age']}")
        print(f"   40yd: {row['Sprint_40yd']}, Vertical: {row['Vertical_Jump']}, Bench: {row['Bench_Press_Reps']}")
        print(f"   Broad Jump: {row['Broad_Jump']}, Shuttle: {row['Shuttle']}, 3Cone: {row['Agility_3cone']}")

# Statistical analysis of surprising cases
print("\n" + "=" * 70)
print("統計的分析: パターンはあるか？")
print("=" * 70)

# Analyze by position
print("\n【ポジション別の誤分類率】")
pos_stats = train.groupby('Position').agg({
    'correct': ['sum', 'count'],
    'actual': 'mean'
}).round(3)
pos_stats.columns = ['correct_count', 'total', 'draft_rate']
pos_stats['error_rate'] = 1 - pos_stats['correct_count'] / pos_stats['total']
pos_stats = pos_stats.sort_values('error_rate', ascending=False)
print(pos_stats.head(15))

# Analyze false positives by position
print("\n【Type 1 (FP): ポジション分布】- 能力高いのにドラフト外")
if len(false_positives) > 0:
    fp_pos = false_positives['Position'].value_counts()
    print(fp_pos)

# Analyze false negatives by position
print("\n【Type 2 (FN): ポジション分布】- 能力低そうなのにドラフト")
if len(false_negatives) > 0:
    fn_pos = false_negatives['Position'].value_counts()
    print(fn_pos)

# School analysis for surprising cases
print("\n【学校別分析】")
print("\nType 1 (FP) - 能力高いのにドラフト外の学校 Top 15:")
if len(false_positives) > 0:
    fp_schools = false_positives['School'].value_counts().head(15)
    print(fp_schools)

    # Check if these schools have lower draft rates in general
    print("\n  → これらの学校の全体ドラフト率:")
    for school in fp_schools.index[:10]:
        school_data = train[train['School'] == school]
        draft_rate = school_data['actual'].mean()
        total = len(school_data)
        print(f"     {school}: {draft_rate:.1%} ({total}人)")

print("\nType 2 (FN) - 能力低そうなのにドラフトされた学校 Top 15:")
if len(false_negatives) > 0:
    fn_schools = false_negatives['School'].value_counts().head(15)
    print(fn_schools)

    # Check if these schools have higher draft rates in general
    print("\n  → これらの学校の全体ドラフト率:")
    for school in fn_schools.index[:10]:
        school_data = train[train['School'] == school]
        draft_rate = school_data['actual'].mean()
        total = len(school_data)
        print(f"     {school}: {draft_rate:.1%} ({total}人)")

# Compare feature distributions
print("\n" + "=" * 70)
print("特徴量の比較: 意外なケース vs 正常なケース")
print("=" * 70)

numeric_features = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump',
                    'Shuttle', 'Agility_3cone', 'Height', 'Weight', 'Age']

print("\n【Type 1 (FP): 能力は高いはずなのにドラフト外】")
print("特徴量の平均比較:")
if len(false_positives) > 0:
    normal_undrafted = train[(train['actual'] == 0) & (train['oof_pred'] < 0.5)]
    print(f"{'特徴量':15} {'FP':>10} {'通常ドラフト外':>15} {'ドラフト済み':>12}")
    print("-" * 55)
    for feat in numeric_features:
        fp_mean = false_positives[feat].mean()
        normal_mean = normal_undrafted[feat].mean()
        drafted_mean = train[train['actual'] == 1][feat].mean()
        # Show if FP is closer to drafted or undrafted
        marker = "★" if abs(fp_mean - drafted_mean) < abs(fp_mean - normal_mean) else ""
        print(f"{feat:15} {fp_mean:10.2f} {normal_mean:15.2f} {drafted_mean:12.2f} {marker}")

print("\n【Type 2 (FN): 能力低そうなのにドラフト】")
print("特徴量の平均比較:")
if len(false_negatives) > 0:
    normal_drafted = train[(train['actual'] == 1) & (train['oof_pred'] >= 0.5)]
    print(f"{'特徴量':15} {'FN':>10} {'通常ドラフト':>12} {'ドラフト外':>12}")
    print("-" * 50)
    for feat in numeric_features:
        fn_mean = false_negatives[feat].mean()
        normal_mean = normal_drafted[feat].mean()
        undrafted_mean = train[train['actual'] == 0][feat].mean()
        # Show if FN is closer to undrafted
        marker = "★" if abs(fn_mean - undrafted_mean) < abs(fn_mean - normal_mean) else ""
        print(f"{feat:15} {fn_mean:10.2f} {normal_mean:12.2f} {undrafted_mean:12.2f} {marker}")

# Missing data analysis
print("\n" + "=" * 70)
print("欠損値分析: 意外なケースにデータ欠損が多い？")
print("=" * 70)

print(f"{'特徴量':20} {'FP欠損率':>10} {'FN欠損率':>10} {'全体欠損率':>10}")
print("-" * 55)
for feat in numeric_features:
    fp_missing = false_positives[feat].isna().mean() if len(false_positives) > 0 else 0
    fn_missing = false_negatives[feat].isna().mean() if len(false_negatives) > 0 else 0
    all_missing = train[feat].isna().mean()
    fp_flag = "↑" if fp_missing > all_missing * 1.2 else ""
    fn_flag = "↑" if fn_missing > all_missing * 1.2 else ""
    print(f"{feat:20} {fp_missing:9.1%}{fp_flag} {fn_missing:9.1%}{fn_flag} {all_missing:10.1%}")

# Year analysis
print("\n" + "=" * 70)
print("年度別分析: 特定の年に偏りがあるか？")
print("=" * 70)

print("\n【Type 1 (FP) 年度分布】")
if len(false_positives) > 0:
    fp_year = false_positives['Year'].value_counts().sort_index()
    total_year = train['Year'].value_counts().sort_index()
    print(f"{'年度':>6} {'FP数':>6} {'全体数':>8} {'FP率':>8}")
    for year in sorted(train['Year'].unique()):
        fp_count = fp_year.get(year, 0)
        total_count = total_year.get(year, 0)
        rate = fp_count / total_count if total_count > 0 else 0
        print(f"{year:>6} {fp_count:>6} {total_count:>8} {rate:>8.1%}")

# Position Type analysis
print("\n" + "=" * 70)
print("Position Type別分析")
print("=" * 70)

print("\n【Type 1 (FP): Position Type分布】")
if len(false_positives) > 0:
    fp_pt = false_positives['Position_Type'].value_counts()
    total_pt = train[train['actual'] == 0]['Position_Type'].value_counts()
    for pt in fp_pt.index:
        fp_count = fp_pt.get(pt, 0)
        total_count = total_pt.get(pt, 0)
        rate = fp_count / total_count if total_count > 0 else 0
        print(f"  {pt}: {fp_count}人 (ドラフト外全体の{rate:.1%})")

print("\n【Type 2 (FN): Position Type分布】")
if len(false_negatives) > 0:
    fn_pt = false_negatives['Position_Type'].value_counts()
    total_pt = train[train['actual'] == 1]['Position_Type'].value_counts()
    for pt in fn_pt.index:
        fn_count = fn_pt.get(pt, 0)
        total_count = total_pt.get(pt, 0)
        rate = fn_count / total_count if total_count > 0 else 0
        print(f"  {pt}: {fn_count}人 (ドラフト全体の{rate:.1%})")

# Conclusion
print("\n" + "=" * 70)
print("結論")
print("=" * 70)
print(f"\n全体の正解率: {train['correct'].mean():.1%}")
print(f"Type 1 (高確信FP): {len(false_positives)}人 - 能力は高いがドラフト外")
print(f"Type 2 (高確信FN): {len(false_negatives)}人 - 能力は低そうだがドラフト")

# Interpretation
print("\n" + "=" * 70)
print("解釈と仮説")
print("=" * 70)
print("""
【Type 1 - 能力高いのにドラフト外】の可能性:
  1. 怪我歴・性格問題など、数値に現れない要因
  2. 特定学校の選手が過小評価されている
  3. コンバイン成績は良いが試合パフォーマンスが低い
  4. ポジション競争が激しい年度だった

【Type 2 - 能力低そうなのにドラフト】の可能性:
  1. 強豪校出身で実績がある（ブランド力）
  2. 特定ポジションで人材不足だった年度
  3. コンバインでは測れない能力がある
  4. 遅咲きの選手（ポテンシャル評価）
""")

# Save for further analysis
train[['Position', 'Position_Type', 'School', 'Year', 'Height', 'Weight', 'Age',
       'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump',
       'Shuttle', 'Agility_3cone', 'actual', 'oof_pred', 'error', 'correct']].to_csv(
    '/home/user/competition2/experiments/exp11_pseudo_label/prediction_analysis.csv', index=False)
print("\n詳細データを prediction_analysis.csv に保存しました")
