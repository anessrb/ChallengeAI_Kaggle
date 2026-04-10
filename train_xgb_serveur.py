import warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA     = Path('/home/barrage/challenge2026MIASHS')
ENV      = DATA / 'EnvironmentalValues'
WORK_DIR = Path('/home/grp4')
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ─── METADATA ─────────────────────────────────────────────────────────────────
pa_train = pd.read_csv(DATA / 'GLC25_PA_metadata_train.csv')
pa_test  = pd.read_csv(DATA / 'GLC25_PA_metadata_test.csv')

train_surveys = pa_train[['surveyId','lon','lat','year','region']].drop_duplicates('surveyId')
test_surveys  = pa_test[['surveyId','lon','lat','year','region']].drop_duplicates('surveyId')

train_labels = (
    pa_train.dropna(subset=['speciesId'])
    .groupby('surveyId')['speciesId']
    .apply(lambda x: list(x.astype(int).unique()))
    .reset_index()
    .rename(columns={'speciesId': 'species_list'})
)
print(f'Train: {len(train_surveys)} | Test: {len(test_surveys)} | Especes: {pa_train["speciesId"].nunique()}')

# ─── FEATURES TABULAIRES ──────────────────────────────────────────────────────
def load_env(tr, te):
    return pd.read_csv(tr), pd.read_csv(te)

elev_tr,    elev_te    = load_env(ENV/'Elevation/GLC25-PA-train-elevation.csv',
                                   ENV/'Elevation/GLC25-PA-test-elevation.csv')
soil_tr,    soil_te    = load_env(ENV/'SoilGrids/GLC25-PA-train-soilgrids.csv',
                                   ENV/'SoilGrids/GLC25-PA-test-soilgrids.csv')
land_tr,    land_te    = load_env(ENV/'LandCover/GLC25-PA-train-landcover.csv',
                                   ENV/'LandCover/GLC25-PA-test-landcover.csv')
foot_tr,    foot_te    = load_env(ENV/'HumanFootprint/GLC25-PA-train-human_footprint.csv',
                                   ENV/'HumanFootprint/GLC25-PA-test-human_footprint.csv')
bioclim_tr, bioclim_te = load_env(ENV/'ClimateAverage_1981-2010/GLC25-PA-train-bioclimatic.csv',
                                   ENV/'ClimateAverage_1981-2010/GLC25-PA-test-bioclimatic.csv')
print('Features tabulaires chargees')

# ─── BIOCLIM TIME SERIES STATS ────────────────────────────────────────────────
print('Chargement BioclimTimeSeries...')
bts_tr = pd.read_csv(DATA / 'BioclimTimeSeries/values/GLC25-PA-train-bioclimatic_monthly.csv')
bts_te = pd.read_csv(DATA / 'BioclimTimeSeries/values/GLC25-PA-test-bioclimatic_monthly.csv')

def compute_ts_stats(df):
    sid      = df[['surveyId']].copy()
    feat_cols = [c for c in df.columns if c != 'surveyId']
    vals     = df[feat_cols]
    stats    = [sid]
    for var in ['pr', 'tas', 'tasmax', 'tasmin']:
        cols = [c for c in feat_cols if f'Bio-{var}_' in c]
        sub  = vals[cols]
        stats.append(sub.mean(axis=1).rename(f'{var}_mean'))
        stats.append(sub.std(axis=1).rename(f'{var}_std'))
        stats.append(sub.min(axis=1).rename(f'{var}_min'))
        stats.append(sub.max(axis=1).rename(f'{var}_max'))
        time_idx = np.arange(len(cols))
        trend = sub.apply(lambda row: np.polyfit(time_idx, row.fillna(row.mean()), 1)[0], axis=1)
        stats.append(trend.rename(f'{var}_trend'))
    return pd.concat(stats, axis=1)

print('Calcul stats train...')
bts_stats_tr = compute_ts_stats(bts_tr)
print('Calcul stats test...')
bts_stats_te = compute_ts_stats(bts_te)
print(f'BioclimTS stats: {bts_stats_tr.shape}')

# ─── MERGE FEATURES ───────────────────────────────────────────────────────────
def merge_all(surveys, dfs):
    result = surveys.copy()
    for df in dfs:
        result = result.merge(df, on='surveyId', how='left')
    return result

X_train_df = merge_all(train_surveys, [elev_tr, soil_tr, land_tr, foot_tr, bioclim_tr, bts_stats_tr])
X_test_df  = merge_all(test_surveys,  [elev_te, soil_te, land_te, foot_te, bioclim_te, bts_stats_te])

reg_tr = pd.get_dummies(X_train_df['region'], prefix='region')
reg_te = pd.get_dummies(X_test_df['region'],  prefix='region').reindex(columns=reg_tr.columns, fill_value=0)
X_train_df = pd.concat([X_train_df.drop(columns=['region']), reg_tr], axis=1)
X_test_df  = pd.concat([X_test_df.drop(columns=['region']),  reg_te], axis=1)

train_df     = X_train_df.merge(train_labels, on='surveyId', how='inner')
feature_cols = [c for c in X_train_df.columns if c != 'surveyId']

X_train = train_df[feature_cols].values.astype(np.float32)
X_test  = X_test_df[feature_cols].values.astype(np.float32)

col_medians = np.nanmedian(X_train, axis=0)
for i in range(X_train.shape[1]):
    X_train[np.isnan(X_train[:, i]), i] = col_medians[i]
    X_test[np.isnan(X_test[:, i]), i]   = col_medians[i]

print(f'X_train: {X_train.shape} | X_test: {X_test.shape}')

# ─── MULTI-LABEL ──────────────────────────────────────────────────────────────
all_species = sorted(pa_train['speciesId'].dropna().astype(int).unique())
mlb = MultiLabelBinarizer(classes=all_species)
Y_train = mlb.fit_transform(train_df['species_list'])
print(f'Y_train: {Y_train.shape}')

# ─── XGBOOST GPU ──────────────────────────────────────────────────────────────
species_counts = pa_train['speciesId'].dropna().astype(int).value_counts()
MIN_OCC = 10
common_species = species_counts[species_counts >= MIN_OCC].index.tolist()
species_to_idx = {s: i for i, s in enumerate(mlb.classes_)}
common_idx     = [species_to_idx[s] for s in common_species if s in species_to_idx]
print(f'Especes a entrainer: {len(common_species)}')

idx_all        = np.arange(len(X_train))
idx_tr, idx_val = train_test_split(idx_all, test_size=0.2, random_state=42)

xgb_params = dict(
    objective='binary:logistic',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    device='cuda',
    eval_metric='logloss',
    verbosity=0,
)

test_probs = np.zeros((len(X_test),       len(all_species)), dtype=np.float32)
val_probs  = np.zeros((len(idx_val),      len(all_species)), dtype=np.float32)

t0 = time.time()
for i, (sp_idx, sp_id) in enumerate(zip(common_idx, common_species)):
    y = Y_train[:, sp_idx]
    if y.sum() < MIN_OCC:
        continue
    clf = XGBClassifier(**xgb_params)
    clf.fit(X_train[idx_tr], y[idx_tr])
    val_probs[:, sp_idx]  = clf.predict_proba(X_train[idx_val])[:, 1]
    test_probs[:, sp_idx] = clf.predict_proba(X_test)[:, 1]
    if (i+1) % 500 == 0:
        elapsed   = (time.time() - t0) / 60
        remaining = elapsed / (i+1) * (len(common_idx) - i - 1)
        print(f'  {i+1}/{len(common_idx)} especes | {elapsed:.1f}min ecoule | ~{remaining:.1f}min restantes')

print(f'XGBoost termine en {(time.time()-t0)/60:.1f}min')

# Sauvegarde des probs pour l'ensemble
np.save(WORK_DIR / 'xgb_test_probs.npy', test_probs)
np.save(WORK_DIR / 'xgb_val_probs.npy',  val_probs)
np.save(WORK_DIR / 'xgb_val_idx.npy',    idx_val)
print('Probs sauvegardees dans /home/grp4/')

# ─── OPTIMISATION SEUIL ───────────────────────────────────────────────────────
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

Y_val = Y_train[idx_val]
best_threshold, best_top_k, best_score = 0.1, 10, 0.0
for threshold in np.arange(0.05, 0.50, 0.025):
    for top_k in [5, 10, 15, 20]:
        score = compute_fscore(val_probs, Y_val, threshold, top_k)
        if score > best_score:
            best_score, best_threshold, best_top_k = score, threshold, top_k

print(f'Meilleur seuil: {best_threshold:.3f} | TOP_K: {best_top_k} | F-score val: {best_score:.4f}')

# ─── SOUMISSION XGB ───────────────────────────────────────────────────────────
predictions = []
for i in range(len(X_test)):
    probs = test_probs[i]
    above = np.where(probs >= best_threshold)[0]
    if len(above) < best_top_k:
        pred_idx = np.union1d(above, np.argsort(probs)[::-1][:best_top_k])
    else:
        pred_idx = above
    predictions.append(' '.join(map(str, sorted([int(mlb.classes_[j]) for j in pred_idx]))))

sub_path = WORK_DIR / 'submission_xgb_serveur.csv'
pd.DataFrame({'surveyId': X_test_df['surveyId'].values, 'predictions': predictions}).to_csv(sub_path, index=False)
print(f'Soumission XGB sauvegardee: {sub_path}')
