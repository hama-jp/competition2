# Exp42: Logit-Mean Ensemble

## Hypothesis
ロジット平均アンサンブルは、異なるキャリブレーションのモデルを混ぜるときに有効。
勾配ブースティング系（LGB/XGB/CAT）だけだと効果は薄いはず。

## Method
- Simple Average: `final = w1*p1 + w2*p2 + w3*p3`
- Logit Average: `final = sigmoid(w1*logit(p1) + w2*logit(p2) + w3*logit(p3))`

## Results

### Individual Model CV
| Model | CV AUC |
|-------|--------|
| LGB | 0.83688 |
| XGB | 0.83612 |
| CAT | 0.84028 |

### Ensemble Comparison
| Method | Best Weights | CV AUC |
|--------|--------------|--------|
| Simple Average | [0.3, 0.0, 0.7] | 0.84219 |
| Logit Average | [0.3, 0.0, 0.7] | 0.84230 |

**Difference: +0.00010** (ほぼ誤差の範囲内)

### Same Weight Comparison
| Weights | Simple | Logit | Diff |
|---------|--------|-------|------|
| [1.0, 0.0, 0.0] | 0.83688 | 0.83688 | +0.00000 |
| [0.0, 1.0, 0.0] | 0.83612 | 0.83612 | +0.00000 |
| [0.0, 0.0, 1.0] | 0.84028 | 0.84028 | +0.00000 |
| [0.3, 0.3, 0.3] | 0.84089 | 0.84079 | -0.00010 |
| [0.2, 0.2, 0.6] | 0.84193 | 0.84199 | +0.00006 |
| [0.3, 0.3, 0.4] | 0.84138 | 0.84135 | -0.00003 |

### Probability Distribution Analysis
```
LGB:  mean=0.6534, std=0.2756, min=0.0055, max=0.9814
XGB:  mean=0.6585, std=0.2881, min=0.0066, max=0.9860
CAT:  mean=0.6382, std=0.2723, min=0.0061, max=0.9901
```

極端な予測の割合:
- LGB: <0.1 = 10.5%, >0.9 = 10.1%
- XGB: <0.1 = 12.3%, >0.9 = 14.0%
- CAT: <0.1 = 12.0%, >0.9 = 6.7%

## Conclusion
**仮説通り、勾配ブースティング系だけだとロジット平均の効果は限定的**

理由:
1. 確率分布が非常に似ている（mean, stdがほぼ同じ）
2. 極端な予測（<0.1, >0.9）の割合も似ている
3. 同系統モデルは確率キャリブレーションが似ている

ロジット平均が効くケース:
- NNとGBDTを混ぜる場合
- 確率の偏りが異なるモデルを混ぜる場合
- Calibration Curveが大きく異なるモデル同士の場合
