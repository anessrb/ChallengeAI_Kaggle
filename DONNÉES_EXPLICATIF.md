# GeoLifeCLEF 2025 — Description complète des données

## Vue d'ensemble

Le challenge consiste à prédire quelles espèces végétales sont présentes à un endroit donné.
Pour chaque `surveyId` (= une localisation GPS à un moment donné), on doit prédire une **liste d'espèces**.

Toutes les tables sont reliées via la clé **`surveyId`**.

---

## 1. Fichiers d'observations (les labels)

### `GLC25_PA_metadata_train.csv` — Données d'entraînement PA
> **PA = Presence-Absence** : des botanistes sont allés sur le terrain et ont listé toutes les espèces présentes ET absentes. C'est de la donnée de qualité.

| Colonne | Type | Exemple | Description |
|---|---|---|---|
| `surveyId` | int | 212 | Identifiant unique du relevé terrain. **Clé de jointure** avec toutes les autres tables. Un même surveyId peut apparaître sur plusieurs lignes (une par espèce présente). |
| `lon` | float | 3.099038 | Longitude GPS (WGS84) du point de relevé |
| `lat` | float | 43.134956 | Latitude GPS (WGS84) du point de relevé |
| `year` | int | 2021 | Année du relevé terrain (2017 à 2021) |
| `geoUncertaintyInM` | float | 5.0 | Incertitude de la position GPS en mètres (5m = très précis) |
| `areaInM2` | float | 100.0 | Surface du relevé en m² (ex: 100m² = carré de 10×10m) |
| `region` | str | MEDITERRANEAN | Région biogéographique européenne. 8 valeurs possibles : MEDITERRANEAN, CONTINENTAL, ATLANTIC, ALPINE, PANNONIAN, BOREAL, STEPPIC, BLACK SEA |
| `country` | str | France | Pays du relevé |
| `speciesId` | int | 6874 | Identifiant de l'espèce présente. **C'est le label à prédire.** Chaque surveyId a autant de lignes qu'il y a d'espèces présentes. |

**Stats clés :**
- 1 483 637 lignes au total
- 88 987 surveys uniques
- 5 016 espèces uniques
- En moyenne ~17 espèces par survey

---

### `GLC25_PA_metadata_test.csv` — Données de test PA
> Même structure que le train, **sans la colonne `speciesId`** (c'est ce qu'on doit prédire).

| Colonne | Description |
|---|---|
| `surveyId` | Identifiant du survey à prédire |
| `lon`, `lat` | Coordonnées GPS |
| `year` | Année du relevé |
| `geoUncertaintyInM` | Incertitude GPS |
| `areaInM2` | Surface du relevé |
| `region` | Région biogéographique |
| `country` | Pays |

**Stats clés :** 14 784 surveys à prédire

---

### `GLC25_P0_metadata_train.csv` — Données d'entraînement PO
> **PO = Presence-Only** : observations citoyennes (appli Pl@ntNet, GBIF...). Pas d'absences confirmées — si une espèce n'est pas listée, on ne sait pas si elle est absente ou juste non observée.

| Colonne | Type | Exemple | Description |
|---|---|---|---|
| `surveyId` | int | 1 | Identifiant de l'observation |
| `publisher` | str | Pl@ntNet | Source de l'observation (Pl@ntNet, iNaturalist, GBIF...) |
| `year` | int | 2019 | Année de l'observation |
| `month` | int | 5 | Mois de l'observation |
| `day` | int | 5 | Jour de l'observation |
| `lat`, `lon` | float | 43.74, 1.57 | Coordonnées GPS |
| `geoUncertaintyInM` | float | 6.0 | Incertitude GPS |
| `taxonRank` | str | SPECIES | Rang taxonomique (SPECIES, GENUS...) |
| `date` | str | 2019-05-05 | Date complète |
| `dayOfYear` | int | 125 | Jour de l'année (1-365) |
| `speciesId` | int | 3383 | Espèce observée |

**Stats clés :** ~5 millions d'observations (non utilisées dans notre baseline — prof a dit PA suffit)

---

### `GLC25_SAMPLE_SUBMISSION.csv` — Format de soumission
> Montre exactement le format attendu pour soumettre sur Kaggle.

| Colonne | Description |
|---|---|
| `surveyId` | ID du survey de test |
| `predictions` | Liste d'espèces séparées par des espaces, **triées par ordre croissant** |

Exemple : `642, 3301 7301 2436 4600 2860 10897`

---

## 2. Variables environnementales (les features)

Toutes ces tables ont **`surveyId`** comme première colonne pour la jointure. Elles sont disponibles en version train et test.

---

### `EnvironmentalValues/Elevation/` — Altitude
> L'altitude est l'un des prédicteurs les plus importants pour les espèces végétales. Les plantes alpines ne poussent pas en plaine et vice-versa.

| Colonne | Unité | Description |
|---|---|---|
| `Elevation` | mètres | Altitude du point de relevé au-dessus du niveau de la mer |

**Source :** ASTER Global Digital Elevation Model V3 (résolution ~30m)

---

### `EnvironmentalValues/SoilGrids/` — Propriétés du sol
> La composition du sol détermine quelles plantes peuvent y survivre (pH, matière organique, texture...).

| Colonne | Unité | Description |
|---|---|---|
| `Soilgrid-bdod` | cg/cm³ | **Densité apparente** — compacité du sol. Sol dense = moins d'air et d'eau disponible |
| `Soilgrid-cec` | mmol/kg | **Capacité d'échange cationique** — capacité du sol à retenir les nutriments (calcium, magnésium...) |
| `Soilgrid-cfvo` | cm³/dm³ | **Fragments grossiers** — proportion de cailloux/graviers dans le sol |
| `Soilgrid-clay` | g/kg | **Teneur en argile** — sol argileux = retient l'eau, imperméable |
| `Soilgrid-nitrogen` | cg/kg | **Teneur en azote** — nutriment essentiel pour les plantes |
| `Soilgrid-phh2o` | pH×10 | **pH du sol** (mesuré dans l'eau). Sol acide (<7) vs basique (>7) — crucial pour les espèces calcicoles/calcifuges |
| `Soilgrid-sand` | g/kg | **Teneur en sable** — sol sableux = drainage rapide, sec |
| `Soilgrid-silt` | g/kg | **Teneur en limon** — intermédiaire entre argile et sable |
| `Soilgrid-soc` | dg/kg | **Carbone organique** — matière organique = fertilité du sol |

**Source :** SoilGrids (résolution ~1km, profondeur 5-15cm)

---

### `EnvironmentalValues/LandCover/` — Occupation du sol
> Type de végétation/terrain autour du point. Une forêt dense vs une prairie ouverte n'accueille pas les mêmes espèces.

| Colonne | Description |
|---|---|
| `LandCover-1` | Forêts de feuillus (Broadleaf forest) |
| `LandCover-2` | Forêts de conifères (Needleleaf forest) |
| `LandCover-3` | Forêts mixtes |
| `LandCover-4` | Arbustes (Shrubland) |
| `LandCover-5` | Herbacées/prairies (Grassland) |
| `LandCover-6` | Terres cultivées (Cropland) |
| `LandCover-7` | Zones urbaines (Urban) |
| `LandCover-8` | Mosaïque cultures/végétation |
| `LandCover-9` | Zones humides (Wetland) |
| `LandCover-10` | Neige/glace permanente |
| `LandCover-11` | Plans d'eau (lacs, rivières) |
| `LandCover-12` | Zones nues (bare soil, désert) |
| `LandCover-13` | Confiance de la classification IGBP |

**Source :** MODIS Terra+Aqua 500m

---

### `EnvironmentalValues/HumanFootprint/` — Empreinte humaine
> Mesure la présence humaine autour du point. Les espèces rares évitent les zones très anthropisées.

| Colonne | Description |
|---|---|
| `HumanFootprint-cemetery` | Présence de cimetière |
| `HumanFootprint-reservoir` | Présence de réservoir/retenue d'eau |
| `HumanFootprint-greenhouse` | Présence de serres agricoles |
| `HumanFootprint-farmland` | Terres agricoles |
| `HumanFootprint-building-copernicus` | Bâtiments (données Copernicus) |
| `HumanFootprint-forest` | Forêts gérées |
| `HumanFootprint-harbour` | Port |
| `HumanFootprint-building-residential` | Bâtiments résidentiels |
| `HumanFootprint-building-commercial` | Bâtiments commerciaux |
| `HumanFootprint-grass` | Pelouses/gazons urbains |
| `HumanFootprint-quarry` | Carrières |
| `HumanFootprint-road` | Routes |
| `HumanFootprint-salt` | Zones salées/salines |
| `HumanFootprint-building-industrial` | Bâtiments industriels |
| `HumanFootprint-vineyard` | Vignobles |
| `HumanFootprint-military` | Zones militaires |
| `HumanFootprint-construction-site` | Chantiers |
| `HumanFootprint-orchard` | Vergers |
| `HumanFootprint-meadow` | Prairies |
| `HumanFootprint-farmyard` | Cours de ferme |
| `HumanFootprint-dump-site` | Décharges |
| `HumanFootprint-railway` | Voies ferrées |

**Source :** OpenStreetMap via Ecodatacube (résolution 10-30m)

---

### `EnvironmentalValues/ClimateAverage_1981-2010/` — Variables bioclimatiques
> Moyennes climatiques sur 30 ans. Ce sont les variables les plus utilisées en modélisation d'espèces (SDM). Elles capturent le régime climatique d'un lieu.

| Colonne | Description complète |
|---|---|
| `Bio1` | Température annuelle moyenne (°C × 10) |
| `Bio2` | Amplitude thermique journalière moyenne (max - min mensuel) |
| `Bio3` | Isothermalité (Bio2/Bio7 × 100) — régularité des températures |
| `Bio4` | Saisonnalité des températures (écart-type × 100) |
| `Bio5` | Température maximale du mois le plus chaud |
| `Bio6` | Température minimale du mois le plus froid |
| `Bio7` | Amplitude thermique annuelle (Bio5 - Bio6) |
| `Bio8` | Température moyenne du trimestre le plus humide |
| `Bio9` | Température moyenne du trimestre le plus sec |
| `Bio10` | Température moyenne du trimestre le plus chaud |
| `Bio11` | Température moyenne du trimestre le plus froid |
| `Bio12` | Précipitations annuelles totales (mm) |
| `Bio13` | Précipitations du mois le plus humide |
| `Bio14` | Précipitations du mois le plus sec |
| `Bio15` | Saisonnalité des précipitations (coefficient de variation) |
| `Bio16` | Précipitations du trimestre le plus humide |
| `Bio17` | Précipitations du trimestre le plus sec |
| `Bio18` | Précipitations du trimestre le plus chaud |
| `Bio19` | Précipitations du trimestre le plus froid |

**Source :** CHELSA (résolution ~1km)

---

## 3. Séries temporelles

### `BioclimTimeSeries/values/GLC25-PA-train-bioclimatic_monthly.csv`
> Évolution mensuelle du climat sur 20 ans (2000-2019). Capture les tendances climatiques, événements extrêmes, changement climatique local.

**Structure :** 913 colonnes
- Colonne 1 : `surveyId`
- Colonnes 2-913 : valeurs mensuelles pour 4 variables × 12 mois × ~20 ans

**Format des colonnes :** `Bio-{variable}_{mois}_{année}`
- Variables : `pr` (précipitations), `tas` (température moy.), `tasmax` (température max.), `tasmin` (température min.)
- Exemple : `Bio-pr_01_2000` = précipitations de janvier 2000

**Usage dans le modèle :** On calcule des statistiques résumées (moyenne, std, tendance) plutôt que d'utiliser les 912 colonnes brutes.

---

### `BioclimTimeSeries/cubes/PA-train/` et `PA-test/`
> Même données que les CSV mais en format **tenseur PyTorch** (`.pt`), prêts à être chargés directement dans un modèle deep learning.

**Format :** tenseur 3D — axes `[VARIABLE, ANNÉE, MOIS]`  
**Usage :** Directement dans un RNN/LSTM ou Transformer temporel

---

### `SateliteTimeSeries-Landsat/`
> Séries temporelles satellite sur 20 ans (hiver 2000 → automne 2020). Capture l'évolution de la végétation saison par saison.

**6 bandes spectrales × 84 saisons :**
- `R` (Rouge), `G` (Vert), `B` (Bleu) — couleurs visibles
- `NIR` (Proche infrarouge) — indicateur de végétation verte
- `SWIR1`, `SWIR2` (Infrarouge à ondes courtes) — humidité, stress hydrique

**Source :** Landsat (résolution 30m)

---

## 4. Images satellite

### `SatelitePatches/`
> Images satellite centrées sur chaque point de relevé. Permet à un CNN de "voir" le terrain.

**Format :** fichiers TIFF 64×64 pixels, 4 bandes (RGB + NIR)  
**Résolution :** 10m/pixel → couverture de 640m × 640m autour du point  
**Source :** Sentinel-2 (Ecodatacube)

**Accès :** chemin basé sur le surveyId
```
surveyId = 3018575 → ./75/85/3018575.tiff
```
(2 derniers chiffres / 2 avant-derniers chiffres / fichier complet)

**Usage :** Entrée d'un CNN (ResNet, EfficientNet...) pour Phase 3 du projet

---

## 5. Résumé — Quoi utiliser pour quoi

| Phase | Données utilisées | Modèle |
|---|---|---|
| Baseline (Phase 1) | Elevation + SoilGrids + LandCover + HumanFootprint + Bioclimatic → **75 features** | LightGBM |
| Phase 2 | + BioclimTimeSeries (stats) + SatelliteTimeSeries (stats) → **~375 features** | LightGBM amélioré |
| Phase 3 | SatellitePatches (images 64×64) | CNN ResNet18 |
| Phase 4 | Tout | Ensemble LightGBM + CNN |

---

## 6. Schéma des jointures

```
surveyId  ←──────────────────────────────────────────────┐
    │                                                      │
    ├── GLC25_PA_metadata_train.csv  (lon, lat, espèces)  │
    ├── Elevation.csv                (1 feature)           │
    ├── SoilGrids.csv                (9 features)          ├─► X_train (88k × 75)
    ├── LandCover.csv                (13 features)         │
    ├── HumanFootprint.csv           (22 features)         │
    ├── Bioclimatic.csv              (19 features)         │
    └── BioclimTimeSeries.csv        (912 colonnes brutes)─┘
```
