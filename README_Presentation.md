# GeoLifeCLEF 2025 — Species Predictor & MCP Assistant
> Prédiction multimodale d'espèces végétales par point GPS, interface conversationnelle IA et évaluation automatique des prédictions.

---

## Architecture générale

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND React                           │
│   Carte Leaflet │ Panel Espèces │ Explicabilité │ Chat IA       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP (port 5001)
┌──────────────────────────▼──────────────────────────────────────┐
│                     API SERVER FastAPI                          │
│   /api/predict  /api/explain  /api/chat  /api/judge             │
│   Bridge : Mistral (MCP SSE) + Gemma 4 (function calling)       │
└──────────┬───────────────────────────────────┬──────────────────┘
           │ SSE (port 8000)                   │ HTTP direct (port 8001)
┌──────────▼───────────────────────────────────▼──────────────────┐
│                      MCP SERVER FastMCP                         │
│   8 tools exposés • Modèle MultimodalEnsemble chargé en mémoire │
│   122 939 945 paramètres • 3 425 espèces • CPU inference        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Le modèle : MultimodalEnsemble

Réseau de neurones **multimodal** entraîné sur **88 987 surveys** dans **29 pays européens** (2017–2021).

### 3 sources de données en entrée

| Modalité | Shape | Contenu |
|----------|-------|---------|
| **Landsat** (séries temporelles) | `[6, 4, 21]` | 6 bandes spectrales × 4 saisons × 21 ans (2000–2021) |
| **Bioclim** (climat mensuel) | `[4, 19, 12]` | 19 variables BIO WorldClim × 12 mois |
| **Sentinel** (image satellite) | `[4, 128, 128]` | Image haute résolution 4 bandes (RGB + NIR) |

### Architecture

```
Landsat  ──► Swin Transformer ──► proj (768→1000) ──┐
Bioclim  ──► Swin Transformer ──► proj (768→1000) ──┼──► concat [3000] ──► FC ──► 3 425 classes
Sentinel ──► Swin Transformer ──► proj (768→1000) ──┘
```

Chaque modalité passe dans un **Swin Transformer** indépendant. Les 3 représentations sont concaténées et passent dans une tête de classification dense (multi-label, sigmoid).

### Entraînement

- **Dataset** : PA-train — 1 483 637 observations, 5 016 espèces (3 425 retenues)
- **Loss** : Binary Cross-Entropy multi-label
- **Durée** : ~50 epochs sur GPU serveur

---

## MCP Server — 8 outils exposés

Le **Model Context Protocol** (Anthropic, 2024) standardise la communication entre un LLM et des outils externes via SSE. Le LLM découvre automatiquement les outils disponibles au démarrage, sans configuration manuelle côté client.

```bash
python mcp_server.py
# Port 8000 : SSE (Mistral via MCPClientSSE)
# Port 8001 : HTTP direct (API FastAPI)
```

### Liste des tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `predict_species` | `lat, lon, top_k` | Prédit les top-K espèces au point GPS le plus proche |
| `get_species_info` | `species_id` | Observations, régions, étendue géographique |
| `get_environmental_data` | `lat, lon` | Élévation, BIO1–19, type de sol |
| `get_nearby_surveys` | `lat, lon, radius_km` | Surveys d'observation dans un rayon |
| `compare_locations` | `lat1, lon1, lat2, lon2` | Comparaison espèces + indice de Jaccard |
| `explain_prediction` | `lat, lon` | Contribution modalités + SHAP bioclim + Grad-CAM |
| `simulate_climate_change` | `lat, lon, bio5_delta, bio12_delta` | Impact +T°C / -précip sur les espèces |
| `evaluate_vs_ground_truth` | `lat, lon, month, top_k` | Compare prédictions vs espèces réellement observées |

---

## Interface React — 4 onglets

```bash
cd frontend && npm run dev   # http://localhost:5174
```

### 🗺 Carte interactive (Leaflet)
- Clic sur n'importe quel point de France/Europe
- Snap automatique sur le **survey test le plus proche** (14 784 disponibles)

### 🌱 Onglet Espèces
- Top-K espèces prédites avec score de confiance
- Bouton "Expliquer" → bascule vers le chat avec la question pré-remplie

### 🔍 Onglet Explicabilité

Trois visualisations générées à la demande :

**1. Contribution par modalité**
- Méthode : ablation — masquer chaque modalité avec des zéros, mesurer la chute de score
- Camembert global + barres groupées par espèce

**2. Grad-CAM (Sentinel)**
- Heatmap sur l'image satellite : quelle zone influence la prédiction
- Dernière couche du Swin Transformer Sentinel

**3. SHAP par permutation (Bioclim)**
- Pour chaque variable BIO1–BIO19 : permuter ses valeurs temporelles → mesurer la chute de score
- Rouge = influence positive sur la prédiction, Bleu = influence négative

### 💬 Onglet Chat IA
- **Mistral** (via MCP SSE) ou **Gemma 4** (via Gemini function calling)
- Affichage riche : tableaux markdown, graphiques interactifs (bar / hbar / pie / radar)
- Périmètre restreint : refuse les questions hors biodiversité végétale
- Bouton **🧑‍⚖️ Juger la prédiction** → LLM as Judge

---

## LLM as Judge

Évaluation qualitative automatique des prédictions en les comparant au ground truth du jeu d'entraînement.

### Pipeline

```
Requête (lat, lon, mois)
        ↓
① Nearest TRAIN survey  →  espèces réellement observées (ground truth)
② Nearest TEST survey   →  inférence modèle  →  espèces prédites
③ bioclim[:, :, month]  →  profil climatique du mois demandé (19 vars BIO)
④ landsat[:, Q2, :]     →  bandes spectrales de la saison (NDVI proxy)
⑤ Métriques : Precision@K, Recall@K, F1, Jaccard
        ↓
LLM Juge (Gemma 4 / Mistral) :
  "Quercus ilex correctement prédit : NDVI=0.52 en avril,
   BIO5=28.4°C et BIO12=620mm confirment régime méditerranéen."
  "Pinus pinaster faux positif : BIO12=620mm incompatible
   avec l'habitat de lande atlantique typique de cette espèce."
```

### Pourquoi BIO5 et BIO12 dans la simulation climatique ?
- **BIO5** (température max mois chaud) : plafond thermique, +3 à +6°C attendus d'ici 2100
- **BIO12** (précipitations annuelles) : premier facteur de distribution des biomes, -10% à -20% prévu en France du Sud

---

## Lancement complet

```bash
# Terminal 1 — MCP Server (charge le modèle ~30s)
source venv_mcp/bin/activate
python mcp_server.py

# Terminal 2 — API Server
source venv_mcp/bin/activate
export MISTRAL_API_KEY=...
export GEMINI_API_KEY=...
python api_server.py

# Terminal 3 — Frontend
cd frontend && npm run dev
```

---

## Données utilisées

| Fichier / Dossier | Contenu |
|-------------------|---------|
| `GLC25_PA_metadata_train.csv` | 1 483 637 obs., 88 987 surveys, lon/lat/région/pays/speciesId |
| `GLC25_PA_metadata_test.csv` | 14 784 surveys à prédire (sans speciesId) |
| `BioclimTimeSeries/cubes/PA-test/` | Tenseurs bioclim mensuels `[4, 19, 12]` |
| `SateliteTimeSeries-Landsat/cubes/PA-test/` | Séries temporelles Landsat `[6, 4, 21]` |
| `SatelitePatches/PA-test/` | Images Sentinel `.tiff` 4 bandes |
| `EnvironmentalValues/` | CSV : élévation, bioclim moyen, sol, couverture terrestre |

---

## Observabilité — LangSmith

Toutes les conversations et appels de tools sont tracés dans **LangSmith** (project : `ChallengeKaggleMiashs`).
Chaque tour = 1 trace avec les sous-spans pour chaque tool MCP appelé.

---

## Questions / Réponses — Présentation

### Sur le modèle

**Q : Pourquoi un Swin Transformer et pas un ResNet classique ?**
Les images Sentinel ont des structures hiérarchiques (haies, bosquets, parcelles) que le Swin-T capture mieux grâce à ses fenêtres d'attention décalées. De plus, réutiliser la même architecture pour les 3 modalités simplifie le code et permet le transfer learning.

**Q : Comment le modèle gère-t-il les données manquantes ?**
`torch.nan_to_num(..., nan=0.0)` à la lecture des tenseurs. Les surveys sans fichier satellite reçoivent un tenseur de zéros. Le modèle a été entraîné avec cette convention donc il est robuste à cette situation.

**Q : Pourquoi 3 425 espèces et pas 5 016 ?**
Les 3 425 espèces retenues sont celles avec suffisamment d'observations dans le jeu d'entraînement. Les 1 591 espèces restantes ont trop peu de surveys pour être apprises de façon fiable par le réseau.

**Q : Le modèle peut-il prédire à n'importe quel point GPS ?**
Non — il prédit uniquement aux 14 784 positions des surveys test pour lesquels les cubes satellites sont disponibles. Un clic sur la carte fait un snap au survey test le plus proche.

---

### Sur le MCP Server

**Q : Pourquoi MCP et pas une simple API REST ?**
MCP standardise la découverte automatique des outils par le LLM. Mistral détecte les 8 outils disponibles au démarrage sans configuration manuelle. C'est un standard émergent (Anthropic, nov. 2024) qui permet d'interchanger les LLM facilement.

**Q : Pourquoi deux ports (8000 et 8001) ?**
Port 8000 = transport SSE pour Mistral (protocole MCP officiel). Port 8001 = routes HTTP directes pour l'API FastAPI qui importe les fonctions Python sans re-charger les 500 Mo du modèle.

**Q : Le modèle tourne-t-il en temps réel ?**
Oui, inférence sur CPU en ~0.3s par prédiction. Le modèle est chargé une seule fois au démarrage (30s) puis reste en mémoire.

---

### Sur le LLM as Judge

**Q : Est-ce que le LLM peut vraiment évaluer la qualité des prédictions ?**
Il évalue la **cohérence écologique**, pas la précision stricte. Il reçoit les données brutes numériques (BIO5, BIO12, NDVI, élévation) et raisonne sur leur compatibilité avec les espèces prédites. C'est complémentaire aux métriques quantitatives (Precision/Recall/F1).

**Q : Pourquoi les métriques sont parfois à 0 ?**
Le survey train le plus proche et le survey test le plus proche peuvent être à plusieurs km l'un de l'autre, avec des compositions en espèces totalement différentes. C'est précisément ce que le LLM juge explique avec les données environnementales.

**Q : Le SHAP est-il du vrai SHAP ?**
C'est une approximation par permutation (Permutation Feature Importance). On permute les valeurs temporelles de chaque variable BIO et on mesure la chute de score. Ce n'est pas du SHAP exact (Shapley values) mais la sémantique est identique : importance = contribution à la prédiction.

---

### Sur l'interface

**Q : Pourquoi le chat refuse certaines questions ?**
Par conception — un assistant de biodiversité qui répond à des recettes de cuisine perd en crédibilité. Le system prompt refuse explicitement les questions hors-périmètre pour les deux LLM (Mistral et Gemma).

**Q : Comment fonctionnent les graphiques dans le chat ?**
Le LLM génère des blocs ` ```chart {...}``` ` avec un JSON structuré (type, data, axes). Le composant React `ChartBlock` parse ce JSON et le rend avec Recharts. Le LLM est guidé par le system prompt pour utiliser cette syntaxe précise.

**Q : Peut-on étendre le système à d'autres pays ?**
Oui, si les cubes satellites (Landsat, Bioclim, Sentinel) sont disponibles pour les surveys test des autres pays. Le modèle a été entraîné sur 29 pays européens mais les cubes test disponibles localement couvrent principalement la France.

**Q : Pourquoi Gemma 4 et Mistral ? Pourquoi pas ChatGPT ?**
Mistral est utilisé via le protocole MCP natif (SSE), ce qui est l'intégration la plus propre pour démontrer le protocole. Gemma 4 (Google) est utilisé via function calling classique comme point de comparaison. ChatGPT ne supporte pas encore le protocole MCP officiellement.
