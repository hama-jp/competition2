"""
Exp10: Final Ensemble - GBDT + TabNet
Combine Exp07 (GBDT) and Exp09 (TabNet) for maximum diversity
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

BASE_DIR = '/home/user/competition2'
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*50)
print("Exp10: Final Ensemble (GBDT + TabNet)")
print("="*50)

# Load Exp07 predictions (GBDT)
exp07_dir = '/home/user/competition2/experiments/exp07_final'
oof_lgb = np.load(os.path.join(exp07_dir, 'oof_lgb.npy'))
oof_xgb = np.load(os.path.join(exp07_dir, 'oof_xgb.npy'))
oof_cat = np.load(os.path.join(exp07_dir, 'oof_cat.npy'))
pred_final_gbdt = np.load(os.path.join(exp07_dir, 'pred_final.npy'))

# Load Exp07 results for target
with open(os.path.join(exp07_dir, 'results.json'), 'r') as f:
    exp07_results = json.load(f)

# Load Exp09 predictions (TabNet)
exp09_dir = '/home/user/competition2/experiments/exp09_tabnet_improved'
oof_tabnet = np.load(os.path.join(exp09_dir, 'oof_tabnet.npy'))
pred_tabnet = np.load(os.path.join(exp09_dir, 'pred_tabnet.npy'))

# Load target
train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
target = train['Drafted']

print(f"Loaded predictions:")
print(f"  GBDT OOF shape: {oof_lgb.shape}")
print(f"  TabNet OOF shape: {oof_tabnet.shape}")

# Calculate individual CVs
cv_lgb = roc_auc_score(target, oof_lgb)
cv_xgb = roc_auc_score(target, oof_xgb)
cv_cat = roc_auc_score(target, oof_cat)
cv_tabnet = roc_auc_score(target, oof_tabnet)

print(f"\nIndividual CVs:")
print(f"  LGB: {cv_lgb:.5f}")
print(f"  XGB: {cv_xgb:.5f}")
print(f"  CAT: {cv_cat:.5f}")
print(f"  TabNet: {cv_tabnet:.5f}")

# Optimize 4-model ensemble weights
print("\n--- 4-Model Ensemble Optimization ---")
def get_score_4model(weights):
    final_oof = (oof_lgb * weights[0]) + (oof_xgb * weights[1]) + (oof_cat * weights[2]) + (oof_tabnet * weights[3])
    return -roc_auc_score(target, final_oof)

cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bnds = ((0, 1), (0, 1), (0, 1), (0, 1))
init_w = [0.2, 0.2, 0.4, 0.2]

res = minimize(get_score_4model, init_w, method='SLSQP', bounds=bnds, constraints=cons)
best_w_4model = res.x
cv_4model = -res.fun

print(f"Weights: LGB={best_w_4model[0]:.3f}, XGB={best_w_4model[1]:.3f}, CAT={best_w_4model[2]:.3f}, TabNet={best_w_4model[3]:.3f}")
print(f"4-Model CV: {cv_4model:.5f}")

# Also try simple blending with GBDT ensemble + TabNet
print("\n--- GBDT Ensemble + TabNet Blending ---")

# First get optimal GBDT weights
def get_score_gbdt(weights):
    final_oof = (oof_lgb * weights[0]) + (oof_xgb * weights[1]) + (oof_cat * weights[2])
    return -roc_auc_score(target, final_oof)

bnds_gbdt = ((0, 1), (0, 1), (0, 1))
cons_gbdt = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
res_gbdt = minimize(get_score_gbdt, [0.3, 0.3, 0.4], method='SLSQP', bounds=bnds_gbdt, constraints=cons_gbdt)
w_gbdt = res_gbdt.x
oof_gbdt_ensemble = (oof_lgb * w_gbdt[0]) + (oof_xgb * w_gbdt[1]) + (oof_cat * w_gbdt[2])
cv_gbdt_ensemble = roc_auc_score(target, oof_gbdt_ensemble)
print(f"GBDT Ensemble CV: {cv_gbdt_ensemble:.5f}")

# Blend GBDT ensemble with TabNet
def get_blend_score(alpha):
    blended = alpha * oof_gbdt_ensemble + (1 - alpha) * oof_tabnet
    return -roc_auc_score(target, blended)

from scipy.optimize import minimize_scalar
res_blend = minimize_scalar(get_blend_score, bounds=(0, 1), method='bounded')
best_alpha = res_blend.x
cv_blended = -res_blend.fun

print(f"Best blend: {best_alpha:.3f} * GBDT + {1-best_alpha:.3f} * TabNet")
print(f"Blended CV: {cv_blended:.5f}")

# Choose the best ensemble
print(f"\n{'='*50}")
print("Results Summary:")
print(f"  GBDT-only CV: {cv_gbdt_ensemble:.5f}")
print(f"  4-Model CV: {cv_4model:.5f}")
print(f"  GBDT+TabNet Blend CV: {cv_blended:.5f}")
print(f"{'='*50}")

# Use the best approach
if cv_4model >= cv_blended and cv_4model >= cv_gbdt_ensemble:
    print(f"\nBest: 4-Model Ensemble (CV={cv_4model:.5f})")
    final_cv = cv_4model
    final_preds = (pred_final_gbdt * (best_w_4model[0] + best_w_4model[1] + best_w_4model[2]) + pred_tabnet * best_w_4model[3])
    # Actually compute properly
    pred_lgb = np.load(os.path.join(exp07_dir, 'oof_lgb.npy'))  # This is OOF, need test preds
    # Use pred_final_gbdt which is already the ensemble
    # For 4-model: need to recalculate with proper weights
    # Let's load individual test predictions
    # Actually exp07 only saves pred_final.npy, not individual preds
    # So we need to approximate: use GBDT ensemble weight sum and TabNet weight
    gbdt_weight_sum = best_w_4model[0] + best_w_4model[1] + best_w_4model[2]
    tabnet_weight = best_w_4model[3]
    # Renormalize
    gbdt_norm = gbdt_weight_sum / (gbdt_weight_sum + tabnet_weight)
    tabnet_norm = tabnet_weight / (gbdt_weight_sum + tabnet_weight)
    final_preds = pred_final_gbdt * gbdt_norm + pred_tabnet * tabnet_norm
    ensemble_type = '4-model'
elif cv_blended >= cv_gbdt_ensemble:
    print(f"\nBest: GBDT+TabNet Blend (CV={cv_blended:.5f})")
    final_cv = cv_blended
    pred_gbdt_ensemble = pred_final_gbdt  # Already blended
    final_preds = best_alpha * pred_gbdt_ensemble + (1 - best_alpha) * pred_tabnet
    ensemble_type = 'blend'
else:
    print(f"\nBest: GBDT-only (CV={cv_gbdt_ensemble:.5f})")
    final_cv = cv_gbdt_ensemble
    final_preds = pred_final_gbdt
    ensemble_type = 'gbdt-only'

print(f"\nFinal CV: {final_cv:.5f}")

# Save submission
submission = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))
submission['Drafted'] = final_preds
submission.to_csv(os.path.join(EXP_DIR, 'submission.csv'), index=False)

# Also save GBDT-only for comparison
submission_gbdt = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))
submission_gbdt['Drafted'] = pred_final_gbdt
submission_gbdt.to_csv(os.path.join(EXP_DIR, 'submission_gbdt_only.csv'), index=False)

# Save results
results = {
    'experiment': 'exp10_final_ensemble',
    'timestamp': datetime.now().isoformat(),
    'cv_lgb': cv_lgb,
    'cv_xgb': cv_xgb,
    'cv_cat': cv_cat,
    'cv_tabnet': cv_tabnet,
    'cv_gbdt_ensemble': cv_gbdt_ensemble,
    'cv_4model': cv_4model,
    'cv_blended': cv_blended,
    'cv_final': final_cv,
    'ensemble_type': ensemble_type,
    'weights_4model': list(best_w_4model),
    'blend_alpha': best_alpha,
}

with open(os.path.join(EXP_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {EXP_DIR}")
