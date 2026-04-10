"""
Ensemble XGBoost + CNN
Lance APRES train_xgb_serveur.py et train_cnn_serveur.py
Charge les probs sauvegardees et genere la soumission ensemble.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
warnings.filterwarnings('ignore')

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA     = Path('/home/barrage/challenge2026MIASHS')
WORK_DIR = Path('/home/grp4')

# ─── LABELS ───────────────────────────────────────────────────────────────────
pa_train = pd.read_csv(DATA / 'GLC25_PA_metadata_train.csv')
pa_test  = pd.read_csv(DATA / 'GLC25_PA_metadata_test.csv')

all_species = sorted(pa_train['speciesId'].dropna().astype(int).unique())
mlb = MultiLabelBinarizer(classes=all_species)
mlb.fit([all_species])

test_ids = pa_test['surveyId'].unique()
print(f'Test surveys: {len(test_ids)}')

# ─── CHARGEMENT PROBS ─────────────────────────────────────────────────────────
xgb_probs = np.load(WORK_DIR / 'xgb_test_probs.npy')
cnn_probs = np.load(WORK_DIR / 'cnn_test_probs.npy')

print(f'XGB probs: {xgb_probs.shape}')
print(f'CNN probs: {cnn_probs.shape}')

# ─── ENSEMBLE (moyenne ponderee) ──────────────────────────────────────────────
# CNN poids plus eleve car meilleur F-score seul
W_XGB = 0.3
W_CNN = 0.7

ensemble_probs = W_XGB * xgb_probs + W_CNN * cnn_probs
print(f'Ensemble probs: {ensemble_probs.shape}')

# ─── OPTIMISATION SEUIL SUR VALIDATION ───────────────────────────────────────
# Utilise les probs de validation XGB (proxy)
xgb_val_probs = np.load(WORK_DIR / 'xgb_val_probs.npy')
xgb_val_idx   = np.load(WORK_DIR / 'xgb_val_idx.npy')

train_labels = (
    pa_train.dropna(subset=['speciesId'])
    .groupby('surveyId')['speciesId']
    .apply(lambda x: list(x.astype(int).unique()))
    .reset_index()
    .rename(columns={'speciesId': 'species_list'})
)
Y_train = mlb.transform(train_labels['species_list'])
Y_val   = Y_train[xgb_val_idx]

def compute_fscore(probs, Y_true, threshold, top_k):
    f_scores = []
    for i in range(len(probs)):
        above = np.where(probs[i] >= threshold)[0]
        if len(above) < top_k:
            pred_idx = np.union1d(above, np.argsort(probs[i])[::-1][:top_k])
        else:
            pred_idx = above
        pred_set = set(pred_idx)
        true_set = set(np.where(Y_true[i] == 1)[0])
        tp = len(pred_set & true_set)
        if tp == 0: f_scores.append(0.0); continue
        p = tp / len(pred_set)
        r = tp / len(true_set)
        f_scores.append(2*p*r/(p+r))
    return np.mean(f_scores)

best_threshold, best_top_k, best_score = 0.1, 10, 0.0
for threshold in np.arange(0.05, 0.50, 0.025):
    for top_k in [5, 10, 15, 20]:
        score = compute_fscore(xgb_val_probs, Y_val, threshold, top_k)
        if score > best_score:
            best_score, best_threshold, best_top_k = score, threshold, top_k

print(f'Meilleur seuil: {best_threshold:.3f} | TOP_K: {best_top_k} | F-score val XGB: {best_score:.4f}')

# ─── SOUMISSIONS ──────────────────────────────────────────────────────────────
def make_submission(probs, survey_ids, threshold, top_k, path):
    predictions = []
    for i in range(len(probs)):
        p = probs[i]
        above = np.where(p >= threshold)[0]
        if len(above) < top_k:
            pred_idx = np.union1d(above, np.argsort(p)[::-1][:top_k])
        else:
            pred_idx = above
        predictions.append(' '.join(map(str, sorted([int(mlb.classes_[j]) for j in pred_idx]))))
    pd.DataFrame({'surveyId': survey_ids, 'predictions': predictions}).to_csv(path, index=False)
    print(f'Sauvegarde: {path}')

# Soumission ensemble
make_submission(ensemble_probs, test_ids, best_threshold, best_top_k,
                WORK_DIR / 'submission_ensemble.csv')

# Soumission CNN seul (avec seuil fixe 0.2 / TOP_K 10)
make_submission(cnn_probs, test_ids, 0.2, 10,
                WORK_DIR / 'submission_cnn_only.csv')

print('Termine!')
print(f'Fichiers generes dans {WORK_DIR}:')
print('  - submission_ensemble.csv  (XGB 30% + CNN 70%)')
print('  - submission_cnn_only.csv  (CNN seul)')
