"""
MCP Server — GeoLifeCLEF Species Predictor
Expose des tools pour prédire les espèces végétales à un point GPS.

Usage:
    python mcp_server.py

Outils disponibles:
    - predict_species(lat, lon, top_k)     : Prédit les espèces au point GPS (XGBoost)
    - get_species_info(species_id)          : Info sur une espèce
    - get_environmental_data(lat, lon)      : Features environnementales au point GPS
    - get_nearby_surveys(lat, lon, radius)  : Surveys proches dans le dataset

TODO (future):
    - predict_species_cnn(lat, lon)         : Prédiction CNN (images satellite)
    - predict_species_ensemble(lat, lon)    : Ensemble XGB + CNN
    - explain_prediction(lat, lon)          : SHAP values (explicabilité)
    - compare_locations(lat1,lon1,lat2,lon2): Comparer deux points GPS
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from mcp.server.fastmcp import FastMCP

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = Path('/Users/anessrb/Desktop/ChallengeKaggleAI')
ENV_DIR  = DATA_DIR / 'EnvironmentalValues'

# ─── CHARGEMENT DES DONNÉES (au démarrage) ────────────────────────────────────
print("Chargement des données...")

pa_train = pd.read_csv(DATA_DIR / 'GLC25_PA_metadata_train.csv')
pa_test  = pd.read_csv(DATA_DIR / 'GLC25_PA_metadata_test.csv')

# Surveys avec coordonnées
train_surveys = pa_train[['surveyId','lon','lat']].drop_duplicates('surveyId')
test_surveys  = pa_test[['surveyId','lon','lat']].drop_duplicates('surveyId')
all_surveys   = pd.concat([train_surveys, test_surveys], ignore_index=True)

# Espèces observées par survey (train)
train_species = (
    pa_train.dropna(subset=['speciesId'])
    .groupby('surveyId')['speciesId']
    .apply(lambda x: list(x.astype(int).unique()))
    .to_dict()
)

# MLBinarizer pour les classes
all_species_ids = sorted(pa_train['speciesId'].dropna().astype(int).unique())
mlb = MultiLabelBinarizer(classes=all_species_ids)
mlb.fit([all_species_ids])

# Probs XGBoost test
xgb_probs = np.load(DATA_DIR / 'xgb_test_probs.npy')
xgb_survey_ids = pa_test['surveyId'].unique()
xgb_probs_dict = {sid: xgb_probs[i] for i, sid in enumerate(xgb_survey_ids)}

# Features environnementales (test)
def load_env_features():
    try:
        elev    = pd.read_csv(ENV_DIR / 'Elevation/GLC25-PA-test-elevation.csv')
        bioclim = pd.read_csv(ENV_DIR / 'ClimateAverage_1981-2010/GLC25-PA-test-bioclimatic.csv')
        soil    = pd.read_csv(ENV_DIR / 'SoilGrids/GLC25-PA-test-soilgrids.csv')
        merged  = test_surveys.merge(elev, on='surveyId', how='left')
        merged  = merged.merge(bioclim, on='surveyId', how='left')
        merged  = merged.merge(soil, on='surveyId', how='left')
        return merged
    except Exception as e:
        print(f"Warning: features env non chargées: {e}")
        return test_surveys

env_features = load_env_features()
print(f"Données chargées: {len(all_surveys)} surveys | {len(all_species_ids)} espèces")

# ─── HELPER : trouver le survey le plus proche ────────────────────────────────
def find_nearest_survey(lat: float, lon: float, surveys_df: pd.DataFrame) -> dict:
    """Trouve le survey le plus proche par distance euclidienne."""
    dists = np.sqrt((surveys_df['lat'] - lat)**2 + (surveys_df['lon'] - lon)**2)
    idx   = dists.idxmin()
    row   = surveys_df.loc[idx]
    dist_km = dists[idx] * 111  # ~111km par degré
    return {
        'surveyId': int(row['surveyId']),
        'lat': float(row['lat']),
        'lon': float(row['lon']),
        'distance_km': round(dist_km, 2)
    }

# ─── MCP SERVER ───────────────────────────────────────────────────────────────
mcp = FastMCP("GeoLifeCLEF Species Predictor")

@mcp.tool()
def predict_species(lat: float, lon: float, top_k: int = 10) -> str:
    """
    Prédit les espèces végétales les plus probables à un point GPS.
    Utilise le modèle XGBoost entraîné sur les données GeoLifeCLEF 2025.

    Args:
        lat: Latitude du point (ex: 45.75)
        lon: Longitude du point (ex: 4.85)
        top_k: Nombre d'espèces à retourner (défaut: 10)

    Returns:
        Liste des espèces prédites avec leur score de confiance.
    """
    # Trouver le survey test le plus proche
    nearest = find_nearest_survey(lat, lon, test_surveys)
    sid = nearest['surveyId']

    # Récupérer les probs XGBoost
    if sid not in xgb_probs_dict:
        return json.dumps({"error": "Survey non trouvé dans les prédictions XGBoost"})

    probs = xgb_probs_dict[sid]
    top_indices = np.argsort(probs)[::-1][:top_k]

    predictions = []
    for idx in top_indices:
        species_id = int(all_species_ids[idx])
        confidence = round(float(probs[idx]), 4)
        predictions.append({
            "species_id": species_id,
            "confidence": confidence
        })

    result = {
        "query": {"lat": lat, "lon": lon},
        "nearest_survey": nearest,
        "model": "XGBoost (95 features environnementales)",
        "top_predictions": predictions
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_species_info(species_id: int) -> str:
    """
    Retourne des informations sur une espèce végétale à partir de son ID.

    Args:
        species_id: L'identifiant numérique de l'espèce (ex: 1234)

    Returns:
        Informations disponibles sur l'espèce dans le dataset.
    """
    # Chercher les occurrences dans le train
    obs = pa_train[pa_train['speciesId'] == species_id]

    if obs.empty:
        return json.dumps({"error": f"Espèce {species_id} non trouvée dans le dataset"})

    surveys_with_species = obs['surveyId'].nunique()
    regions = obs['region'].dropna().value_counts().to_dict() if 'region' in obs.columns else {}
    lats = obs.merge(train_surveys, on='surveyId', how='left')['lat'].dropna()
    lons = obs.merge(train_surveys, on='surveyId', how='left')['lon'].dropna()

    result = {
        "species_id": species_id,
        "observations_count": len(obs),
        "surveys_count": surveys_with_species,
        "regions": {str(k): int(v) for k, v in list(regions.items())[:5]},
        "geographic_range": {
            "lat_min": round(float(lats.min()), 4) if len(lats) > 0 else None,
            "lat_max": round(float(lats.max()), 4) if len(lats) > 0 else None,
            "lon_min": round(float(lons.min()), 4) if len(lons) > 0 else None,
            "lon_max": round(float(lons.max()), 4) if len(lons) > 0 else None,
        },
        "note": "Pour le nom scientifique, consulter GBIF avec cet ID."
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_environmental_data(lat: float, lon: float) -> str:
    """
    Retourne les données environnementales (élévation, bioclimat, sol) au point GPS le plus proche.

    Args:
        lat: Latitude du point
        lon: Longitude du point

    Returns:
        Features environnementales disponibles au point GPS.
    """
    nearest = find_nearest_survey(lat, lon, test_surveys)
    sid = nearest['surveyId']

    row = env_features[env_features['surveyId'] == sid]
    if row.empty:
        return json.dumps({"error": "Données environnementales non disponibles"})

    row_dict = row.iloc[0].dropna().to_dict()
    # Garder seulement les features numériques utiles
    env_data = {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                for k, v in row_dict.items()
                if k not in ['surveyId', 'lat', 'lon']}

    result = {
        "query": {"lat": lat, "lon": lon},
        "nearest_survey": nearest,
        "environmental_features": env_data
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_nearby_surveys(lat: float, lon: float, radius_km: float = 50.0, max_results: int = 10) -> str:
    """
    Retourne les surveys d'observation les plus proches d'un point GPS.

    Args:
        lat: Latitude du point
        lon: Longitude du point
        radius_km: Rayon de recherche en km (défaut: 50)
        max_results: Nombre maximum de résultats (défaut: 10)

    Returns:
        Liste des surveys proches avec les espèces observées.
    """
    dists = np.sqrt((all_surveys['lat'] - lat)**2 + (all_surveys['lon'] - lon)**2)
    dist_km = dists * 111
    mask = dist_km <= radius_km
    nearby = all_surveys[mask].copy()
    nearby['distance_km'] = dist_km[mask].values
    nearby = nearby.nsmallest(max_results, 'distance_km')

    results = []
    for _, row in nearby.iterrows():
        sid = int(row['surveyId'])
        entry = {
            "surveyId": sid,
            "lat": round(float(row['lat']), 4),
            "lon": round(float(row['lon']), 4),
            "distance_km": round(float(row['distance_km']), 2),
        }
        if sid in train_species:
            entry["observed_species"] = train_species[sid]
            entry["source"] = "train (espèces connues)"
        else:
            entry["source"] = "test (espèces à prédire)"
        results.append(entry)

    result = {
        "query": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "surveys_found": len(results),
        "surveys": results
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ─── TOOLS FUTURS (stubs documentés) ──────────────────────────────────────────

@mcp.tool()
def predict_species_cnn(lat: float, lon: float, top_k: int = 10) -> str:
    """
    [FUTUR] Prédit les espèces via le modèle CNN ResNet18 sur images satellite 64x64.
    À implémenter quand le modèle CNN de l'équipe sera intégré.

    Args:
        lat: Latitude
        lon: Longitude
        top_k: Nombre d'espèces à retourner
    """
    return json.dumps({
        "status": "not_implemented",
        "message": "CNN non encore intégré. Utilise predict_species() (XGBoost) pour l'instant.",
        "todo": "Charger cnn_test_probs.npy + mapper sur les surveyIds"
    })


@mcp.tool()
def explain_prediction(lat: float, lon: float) -> str:
    """
    [FUTUR] Explique la prédiction via SHAP values — quelles features environnementales
    ont le plus influencé la prédiction des espèces à ce point GPS.

    Args:
        lat: Latitude
        lon: Longitude
    """
    return json.dumps({
        "status": "not_implemented",
        "message": "Explicabilité SHAP non encore implémentée.",
        "todo": "Calculer shap.TreeExplainer sur le modèle XGBoost + retourner top features"
    })


@mcp.tool()
def compare_locations(lat1: float, lon1: float, lat2: float, lon2: float, top_k: int = 10) -> str:
    """
    [FUTUR] Compare les prédictions d'espèces entre deux points GPS.
    Retourne espèces communes, espèces uniques à chaque point.

    Args:
        lat1, lon1: Premier point GPS
        lat2, lon2: Deuxième point GPS
        top_k: Nombre d'espèces par point
    """
    return json.dumps({
        "status": "not_implemented",
        "message": "Comparaison de locations non encore implémentée.",
        "todo": "Appeler predict_species() sur les deux points et comparer les sets"
    })


# ─── LANCEMENT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Serveur MCP GeoLifeCLEF démarré !")
    print("Tools disponibles:")
    print("  ✓ predict_species(lat, lon, top_k)")
    print("  ✓ get_species_info(species_id)")
    print("  ✓ get_environmental_data(lat, lon)")
    print("  ✓ get_nearby_surveys(lat, lon, radius_km)")
    print("  ○ predict_species_cnn         [FUTUR]")
    print("  ○ explain_prediction          [FUTUR]")
    print("  ○ compare_locations           [FUTUR]")
    mcp.run()
