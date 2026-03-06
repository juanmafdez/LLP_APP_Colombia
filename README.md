# LLP_APP_Colombia

Code, boundary assets, and experiment notebooks for the paper:

**SwAV-Pretrained Learning from Label Proportions for Scale-Robust 
Land Cover Estimation in Colombia**  
Juan Manuel Fernández Ospina · Raúl Ramos-Pollán  
University of Antioquia, Colombia

---

## Overview

This repository implements a two-stage pipeline that combines 
self-supervised pretraining (SwAV) with Learning from Label Proportions 
(LLP) for vegetation cover estimation from Sentinel-2 RGB median 
composites over Colombian territory.

The pipeline requires only municipality-level vegetation proportions 
as supervision — derived from ESA WorldCover 2020 — without any 
per-chip annotations. A SwAV encoder is pretrained on unlabeled imagery 
from Caldas, Boyacá, and Magdalena, then used to initialize an LLP model 
trained on Quindío and Tolima.

---

## Repository Structure
```
LLP_APP_Colombia/
├── swav/                   # Architecture modules (ResNet backbone,
│                           # multicrop dataset, projection head)
├── notebooks/              # Training and evaluation notebooks
│   ├── data_extraction_sentinel.ipynb
│   ├── data_extraction_esawc.ipynb
│   ├── swav_pretraining_model.ipynb
│   ├── llp_finetuning_model.ipynb
│   └── stats_and_plots.ipynb
├── data_extract/           # Data extraction utilities
├── deptdata/               # WKT/SHP boundary files for Colombian
│                           # departments and municipalities
└── README.md
```

---

## Pipeline

**1. Data preparation** (`data_extraction_sentinel.ipynb`, 
`data_extraction_esawc.ipynb`)  
Sentinel-2 RGB median composites for 2020 are downloaded via Google 
Earth Engine using geetiles. Imagery is tiled into 100×100-pixel chips 
(≈1 km² at 10 m GSD). Vegetation cover proportions are derived from 
ESA WorldCover 2020 and assigned to chips and municipality bags via 
spatial intersection.

**2. SwAV pretraining** (`swav_pretraining_model.ipynb`)  
A ResNet backbone is pretrained with SwAV on unlabeled imagery from 
Caldas, Boyacá, and Magdalena. Training uses a multi-crop strategy 
(2 high-resolution + 3 low-resolution views per image) and proceeds 
in two stages: 100 prototypes for 50 epochs, then 50 prototypes for 
25 additional epochs.

**3. LLP fine-tuning** (`llp_finetuning_model.ipynb`)  
The pretrained backbone initializes an LLP model (encoder + single 
sigmoid head) trained end-to-end on Quindío and Tolima. Two bag 
protocols are supported: municipality-level bags and an incremental 
geo-referenced grid spanning 1 to 4096 km² per bag.

**4. Statistics and plots** (`stats_and_plots.ipynb`)  
Aggregates results across seeds and offsets, computes convergence 
speed metrics (E90, E95, nAUCr), runs Welch t-tests and one-way 
ANOVAs with Benjamini-Hochberg FDR correction, and generates all 
figures reported in the paper.

---

## Key Results

| Setting | SwAV r | Scratch r |
|---------|--------|-----------|
| Tolima grids 1–256 km² | 0.916–0.941 | 0.865–0.922 |
| Quindío municipalities | 0.886–0.939 | −0.951–0.800 |

SwAV reaches 90% of its final Pearson r in 2–3 epochs versus 5–6 
for scratch initialization. In Quindío, where only 9 municipalities 
are available, scratch produces optimization failure in one of two 
seeds (r = −0.951); SwAV initializes stably across both.

---

## Requirements

Experiments were conducted on Google Colaboratory (GPU). 
Dependencies include TensorFlow, GeoPandas, geetiles, and 
standard scientific Python libraries. Imagery and checkpoints 
are stored on Google Drive; paths are configured within each notebook.

---

## Data

- Sentinel-2 RGB composites: Google Earth Engine 
  (`sentinel2-rgb-median-2020` via geetiles)
- Land cover labels: ESA WorldCover 10 m 2020 v100
- Municipal boundaries: IGAC official SHP layers
- Experiment results: stored as CSV files with `interim` 
  and `final` stage flags for full traceability

---

## Citation

If you use this code or the boundary assets, please cite:
```
Fernández Ospina, J. M., & Ramos-Pollán, R. (2025).
SwAV-Pretrained Learning from Label Proportions for 
Scale-Robust Land Cover Estimation in Colombia.
International Journal of Interactive Multimedia and 
Artificial Intelligence (IJIMAI).
```

---

## Acknowledgments

This work was supported by the University of Antioquia through 
a graduate research fellowship awarded to the first author.
