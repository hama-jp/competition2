# Exp43: NN + GBDT Ensemble with Logit Average

## Hypothesis
NNとGBDTは確率キャリブレーションが異なるため、ロジット平均アンサンブルが効くはず。

## Results

### Individual Models
| Model | CV AUC |
|-------|--------|
| CatBoost | **0.84028** |
| MLP(128,64) | 0.80871 |
| MLP(64,32) | 0.81097 |
| MLP(128,64,32) | 0.80372 |
| MLP(128,64)_reg | 0.80998 |

→ MLP単体はCatBoostより大幅に低い

### Ensemble Effect
| Method | Weight (CAT:MLP) | CV AUC | vs CAT only |
|--------|------------------|--------|-------------|
| CatBoost only | 1.0:0.0 | 0.84028 | - |
| Simple Average | 0.70:0.30 | **0.84392** | +0.00364 |
| Logit Average | 0.75:0.25 | **0.84425** | +0.00397 |

→ **MLPは単体で弱いが、アンサンブルで価値あり！**

### Why Logit Average Works

キャリブレーションの違い:
```
CAT:  mean=0.6382, std=0.2723, <0.1=12.0%, >0.9=6.7%
MLP:  mean=0.6668, std=0.2720, <0.1=5.2%, >0.9=21.7%
```

- MLPは「自信過剰」傾向（>0.9が21.7% vs CATの6.7%）
- この違いがロジット平均の効果を生む
- 相関: 0.8151（多様性あり）

### Comparison with Exp42 (GBDT only)
| Experiment | Models | Logit Improvement |
|------------|--------|-------------------|
| exp42 | LGB + XGB + CAT | +0.00010 |
| **exp43** | **CAT + MLP** | **+0.00033** |

→ 異なるファミリーのモデルを混ぜると効果3倍

## Conclusion
1. **MLPは単体で使う価値はない** (0.81 vs 0.84)
2. **しかしアンサンブル多様性として価値あり** (+0.00364~0.00397)
3. **ロジット平均はNN+GBDTで効果が大きい** (GBDT同士の3倍)
4. キャリブレーションの違いが鍵

## Next Steps
- PyTorchでカスタムNN（スキップ接続あり）を試す
- TabNet等の専用アーキテクチャを検討
- LBでの検証
