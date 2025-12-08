"""
Exp20: Post-processing Techniques
- Rank transformation
- Isotonic regression calibration
- Platt scaling
- Power transformation
"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

EXP_DIR = '/home/user/competition2/experiments/exp20_postprocessing'

print("=" * 60)
print("Exp20: Post-processing Techniques")
print("=" * 60)

# Load exp13 predictions (best LB)
train = pd.read_csv('/home/user/competition2/train.csv')
test = pd.read_csv('/home/user/competition2/test.csv')

# Load OOF predictions from exp14 (30 features, 10 seeds - most stable)
oof_lgb = np.load('/home/user/competition2/experiments/exp14_30feat_10seeds/oof_lgb.npy')
oof_xgb = np.load('/home/user/competition2/experiments/exp14_30feat_10seeds/oof_xgb.npy')
oof_cat = np.load('/home/user/competition2/experiments/exp14_30feat_10seeds/oof_cat.npy')
pred_final = np.load('/home/user/competition2/experiments/exp14_30feat_10seeds/pred_final.npy')

# Ensemble OOF
oof_ensemble = (oof_lgb * 0.3 + oof_xgb * 0.3 + oof_cat * 0.4)
y_train = train['Drafted'].values

print(f"OOF shape: {oof_ensemble.shape}")
print(f"Test pred shape: {pred_final.shape}")
print(f"Baseline CV: {roc_auc_score(y_train, oof_ensemble):.5f}")

# ==========================================
# Post-processing Methods
# ==========================================

def rank_transform(oof, pred):
    """Transform predictions to percentile ranks"""
    # Combine all predictions for ranking
    all_preds = np.concatenate([oof, pred])
    ranks = stats.rankdata(all_preds, method='average') / len(all_preds)
    oof_rank = ranks[:len(oof)]
    pred_rank = ranks[len(oof):]
    return oof_rank, pred_rank

def isotonic_calibration(oof, pred, y):
    """Isotonic regression calibration"""
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(oof, y)
    oof_calib = ir.predict(oof)
    pred_calib = ir.predict(pred)
    return oof_calib, pred_calib

def platt_scaling(oof, pred, y):
    """Platt scaling (logistic regression calibration)"""
    lr = LogisticRegression(C=1.0, random_state=42)
    lr.fit(oof.reshape(-1, 1), y)
    oof_calib = lr.predict_proba(oof.reshape(-1, 1))[:, 1]
    pred_calib = lr.predict_proba(pred.reshape(-1, 1))[:, 1]
    return oof_calib, pred_calib

def power_transform(oof, pred, power=0.5):
    """Power transformation"""
    oof_trans = np.power(oof, power)
    pred_trans = np.power(pred, power)
    return oof_trans, pred_trans

def clip_extreme(oof, pred, lower=0.01, upper=0.99):
    """Clip extreme predictions"""
    oof_clip = np.clip(oof, lower, upper)
    pred_clip = np.clip(pred, lower, upper)
    return oof_clip, pred_clip

# ==========================================
# Apply Post-processing
# ==========================================
results = {}

# Baseline
cv_baseline = roc_auc_score(y_train, oof_ensemble)
results['Baseline'] = {'cv': cv_baseline, 'pred': pred_final}
print(f"\nBaseline: CV = {cv_baseline:.5f}")

# Method 1: Rank Transform
print("\n--- Rank Transform ---")
oof_rank, pred_rank = rank_transform(oof_ensemble, pred_final)
cv_rank = roc_auc_score(y_train, oof_rank)
print(f"CV: {cv_rank:.5f}")
results['Rank'] = {'cv': cv_rank, 'pred': pred_rank}

# Method 2: Isotonic Calibration
print("\n--- Isotonic Calibration ---")
oof_iso, pred_iso = isotonic_calibration(oof_ensemble, pred_final, y_train)
cv_iso = roc_auc_score(y_train, oof_iso)
print(f"CV: {cv_iso:.5f}")
results['Isotonic'] = {'cv': cv_iso, 'pred': pred_iso}

# Method 3: Platt Scaling
print("\n--- Platt Scaling ---")
oof_platt, pred_platt = platt_scaling(oof_ensemble, pred_final, y_train)
cv_platt = roc_auc_score(y_train, oof_platt)
print(f"CV: {cv_platt:.5f}")
results['Platt'] = {'cv': cv_platt, 'pred': pred_platt}

# Method 4: Power Transform (various powers)
print("\n--- Power Transform ---")
for power in [0.3, 0.5, 0.7, 1.5, 2.0]:
    oof_pow, pred_pow = power_transform(oof_ensemble, pred_final, power)
    cv_pow = roc_auc_score(y_train, oof_pow)
    print(f"  Power={power}: CV = {cv_pow:.5f}")
    if power == 0.5:
        results['Power_0.5'] = {'cv': cv_pow, 'pred': pred_pow}

# Method 5: Clip Extreme
print("\n--- Clip Extreme ---")
for lower, upper in [(0.01, 0.99), (0.05, 0.95), (0.1, 0.9)]:
    oof_clip, pred_clip = clip_extreme(oof_ensemble, pred_final, lower, upper)
    cv_clip = roc_auc_score(y_train, oof_clip)
    print(f"  Clip ({lower}, {upper}): CV = {cv_clip:.5f}")
    if lower == 0.05:
        results['Clip_0.05'] = {'cv': cv_clip, 'pred': pred_clip}

# Method 6: Blend of transforms
print("\n--- Blend: 0.5*Original + 0.5*Rank ---")
oof_blend = 0.5 * oof_ensemble + 0.5 * oof_rank
pred_blend = 0.5 * pred_final + 0.5 * pred_rank
cv_blend = roc_auc_score(y_train, oof_blend)
print(f"CV: {cv_blend:.5f}")
results['Blend_Rank'] = {'cv': cv_blend, 'pred': pred_blend}

# ==========================================
# Results Summary
# ==========================================
print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)

for name, res in sorted(results.items(), key=lambda x: -x[1]['cv']):
    diff = res['cv'] - cv_baseline
    sign = "+" if diff >= 0 else ""
    print(f"  {name:20}: CV = {res['cv']:.5f} ({sign}{diff:.5f})")

# Find best
best_name = max(results, key=lambda x: results[x]['cv'])
best_cv = results[best_name]['cv']
best_pred = results[best_name]['pred']

print(f"\nBest: {best_name} with CV = {best_cv:.5f}")
print(f"Exp13 reference: CV = 0.85138, LB = 0.84524")

# Note about rank transform
print("\n" + "=" * 60)
print("Note on Rank Transform")
print("=" * 60)
print("Rank transform maintains the same AUC (order is preserved).")
print("It may help with LB if the test distribution differs from train.")
print("Consider submitting 'Rank' version for potential LB improvement.")

# Save submissions
for name, res in results.items():
    submission = pd.DataFrame({
        'Id': test['Id'],
        'Drafted': res['pred']
    })
    submission.to_csv(f'{EXP_DIR}/submission_{name}.csv', index=False)

# Save best
submission = pd.DataFrame({
    'Id': test['Id'],
    'Drafted': best_pred
})
submission.to_csv(f'{EXP_DIR}/submission.csv', index=False)

# Save results
results_summary = {
    'postprocessing_results': {k: v['cv'] for k, v in results.items()},
    'best_method': best_name,
    'best_cv': best_cv,
    'exp13_cv': 0.85138,
    'exp13_lb': 0.84524
}
with open(f'{EXP_DIR}/results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to {EXP_DIR}")
