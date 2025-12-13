# 解法説明書 (Solution Description)
Hitoshi_Takahama

## 1. 概要 (Overview)
本ソリューションでは、データの質を高める**特徴量エンジニアリング**と、堅牢な**検証戦略 (Validation Strategy)** に重点を置きました。
特に、以下の2点がスコア向上の鍵となりました。
1.  **Leak-Free Pipeline**: Cross-Validation (CV) ループ内でTarget Encodingを行うことで、学習データへのリークを完全に防ぎ、信頼できるCVスコアを確立しました。
2.  **Unsupervised Features**: 身体能力データに対する教師なし学習（K-Meansクラスタリング、PCA）を行い、選手の「アーキタイプ（典型的なタイプ）」を特徴量として追加しました。

## 2. 特徴量エンジニアリング (Feature Engineering)

### 2.1 基本特徴量
- `exp33` で有効性が確認された31個の精選された特徴量を使用。
- 身体能力の比率（BMI, Momentum, Speed_Score等）や、ポジションごとの偏差値（Z-Score, Diff）を含みます。

### 2.2 Target Encoding (Leak-Free)
- カテゴリ変数 (`School`, `Position`, `Position_Type`) に対してTarget Encodingを適用。
- **重要**: 「学習前の一括変換」ではなく、**CVのFoldごとに**変換を行うことでリークを排除しました。
    - Trainデータ: Out-of-Fold (OOF) で計算。
    - Valid/Testデータ: Trainデータの統計量を使用。

### 2.3 教師なし学習特徴量 (Unsupervised Features)
選手の身体的特徴（身長、体重、40yd走、垂直跳び等 8項目）を入力とし、以下の特徴量を生成しました。これらは、単独の数値だけでは見えない「選手のタイプ」をモデルに提供します。
- **K-Means Clustering**: 
    - `k=[3, 6, 12, 50]` の4パターンでクラスタリングを行い、所属クラスタIDを特徴量化。
    - 例えば、「大型パワー型」「小型スピード型」のような分類を自動学習。
- **PCA (主成分分析)**:
    - 身体データの分散を説明する主成分 (`n_components=[2, 5]`) を抽出。

※ これらもCVループ内で学習・変換を行いました（厳密なLeak-Free）。

## 3. モデル構成 (Modeling)

以下の3つの強力な勾配ブースティング決定木 (GBDT) モデルのアンサンブルを採用しました。

1.  **CatBoost Classifier**:
    - カテゴリ変数の扱いに長けており、今回の最重要モデル。
    - `Depth=3` と浅めの木を採用し、過学習を抑制。
2.  **LightGBM Classifier**:
    - 高速かつ高精度。`num_leaves=126`, `Depth=3`。
3.  **XGBoost Classifier**:
    - `Depth=5`。他の2つとは異なる特性を持ち、アンサンブルの多様性に貢献。

**アンサンブル手法**:
- 3つのモデルの予測確率を加重平均。
- 重み: `CatBoost (0.4) + LightGBM (0.3) + XGBoost (0.3)`

## 4. 検証 (Validation)
- **手法**: Stratified K-Fold (5分割) × 5 Seeds (計25モデルの平均)
- **評価指標**: ROC AUC
- **結果**:
    - CV Score: **0.84845**
    - Public LB: **0.85144**
    - CVとLBが非常に高く相関しており、信頼性の高いモデルとなっていると思われます。
