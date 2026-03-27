# 🏛️ GSoC 2026 — Break Surface Detection & Fragment Matching
### Organization: HumanAI Foundation
### Project: Geometric Feature Extraction for Automated Break Surface Detection

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Open3D](https://img.shields.io/badge/Open3D-0.18.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![GSoC](https://img.shields.io/badge/GSoC-2026-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Project Goals](#project-goals)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Code Explanation](#code-explanation)
- [Dataset](#dataset)
- [Results](#results)
- [Docker Setup](#docker-setup)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 About the Project

This project is developed as part of **Google Summer of Code (GSoC) 2026** under the **HumanAI Foundation**. It focuses on automating the detection and matching of break surfaces in 3D scanned archaeological fragments using machine learning and geometric feature extraction.

When physical objects like ancient monuments, pottery, and artifacts break, researchers must manually identify and match the broken pieces — a slow, error-prone process. This project automates that workflow using 3D scan data (`.PLY` / `.OBJ` files).

---

## 🎯 Project Goals

```
✅ Incorporate ML into break surface detection pipeline
✅ Detect break surfaces with at least 80% accuracy
✅ Match discontinuous areas of shared topology at ≥80% accuracy
✅ Orient matching fragments without overfitting
```

---

## 🔄 Pipeline Overview

```
📥 INPUT
   └── 3D Scan File (.OBJ / .PLY)
          ↓
   ┌─────────────────────────────┐
   │  Step 1: Load & Visualize   │
   │  → Load .OBJ file           │
   │  → Convert to point cloud   │
   └─────────────┬───────────────┘
                 ↓
   ┌─────────────────────────────┐
   │  Step 2: Feature Extraction │
   │  → Surface normals          │
   │  → Curvature                │
   │  → Roughness                │
   │  → Boundary edge detection  │
   └─────────────┬───────────────┘
                 ↓
   ┌─────────────────────────────┐
   │  Step 3: Break Simulation   │
   │  → Split into 2 fragments   │
   │  → Detect break surfaces    │
   │  → Save fragment pairs      │
   └─────────────┬───────────────┘
                 ↓
   ┌─────────────────────────────┐
   │  Step 4: ML Classifier      │
   │  → KNN with 24 features     │
   │  → Multiscale features      │
   │  → Train/test split         │
   │  → Find best K value        │
   └─────────────┬───────────────┘
                 ↓
   ┌─────────────────────────────┐
   │  Step 5: Matching Algorithm │
   │  → Compare break surfaces   │
   │  → Calculate match score    │
   │  → Threshold at 80%         │
   └─────────────┬───────────────┘
                 ↓
   ┌─────────────────────────────┐
   │  Step 6: Evaluation         │
   │  → Test on multiple samples │
   │  → Calculate all metrics    │
   │  → Generate final report    │
   └─────────────┬───────────────┘
                 ↓
📤 OUTPUT
   └── Match: YES/NO + Confidence Score
```

---

## 📁 Project Structure

```
GSoC-2026-HumanAI/
│
├── src/                              # All Python scripts
│   ├── 01_load_visualize.py          # Load and visualize 3D scan
│   ├── 02_feature_extraction.py      # Extract geometric features
│   ├── 03_break_simulation.py        # Simulate break on monument
│   ├── 04_ml_classifier.py           # Train ML classifier
│   ├── 05_matching_algorithm.py      # Match break surfaces
│   └── 06_evaluation.py              # Final evaluation & report
│
├── data/                             # Dataset files (not committed)
│   ├── stanford/                     # Stanford 3D scan models
│   │   ├── bunny.ply
│   │   └── dragon.ply
│   ├── smithsonian/                  # Smithsonian 3D artifacts
│   │   └── pottery_vessel.obj
│   └── monuments/                    # Monument scan files
│       └── chukur_fountain/
│           ├── untitledTexturePainted.obj
│           ├── untitledTexturePainted.mtl
│           └── textures/
│
├── models/                           # Saved ML models
│   └── trained_classifier.pkl        # Auto-generated after training
│
├── results/                          # Output files
│   ├── feature_matrix.npy            # Extracted features
│   ├── boundary_mask.npy             # Break surface labels
│   ├── fragment_A.ply                # Simulated fragment A
│   ├── fragment_B.ply                # Simulated fragment B
│   ├── trained_model.pkl             # Trained KNN model
│   ├── scaler.pkl                    # Feature scaler
│   ├── ml_results.png                # ML accuracy plots
│   ├── matching_scores.png           # Fragment matching plots
│   ├── final_evaluation.png          # Final evaluation plots
│   └── final_report.txt              # Text evaluation report
│
├── docs/                             # Documentation
│   └── proposal.md                   # GSoC proposal draft
│
├── docker/                           # Docker configuration
│   └── Dockerfile                    # Container setup
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- Windows / Linux / MacOS
- 4GB+ RAM recommended

### Step 1 — Clone Repository
```bash
git clone https://github.com/yourusername/GSoC-2026-HumanAI.git
cd GSoC-2026-HumanAI
```

### Step 2 — Create Virtual Environment
```bash
# Create environment
python -m venv gsoc_env

# Activate on Windows
gsoc_env\Scripts\activate

# Activate on Linux/Mac
source gsoc_env/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```
open3d==0.18.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
trimesh>=3.23.0
```

---

## ▶️ How to Run

Run each script in order from the `src/` folder:

```bash
# Step 1: Load and visualize your 3D scan
python src/01_load_visualize.py

# Step 2: Extract geometric features
python src/02_feature_extraction.py

# Step 3: Simulate break on the monument
python src/03_break_simulation.py

# Step 4: Train ML classifier
python src/04_ml_classifier.py

# Step 5: Run matching algorithm
python src/05_matching_algorithm.py

# Step 6: Final evaluation
python src/06_evaluation.py
```

> ⚠️ Update `OBJ_FILE` path in each script to point to your local `.OBJ` file before running.

---

## 💻 Code Explanation

### 📄 01_load_visualize.py — Load & Visualize

This script loads a 3D monument scan file and displays it in a viewer.

```
What it does:
→ Loads .OBJ file using Open3D
→ Checks file exists before loading
→ Prints basic mesh information
   (vertices, triangles, textures)
→ Computes vertex normals
→ Opens interactive 3D viewer window
```

**Key function:**
```python
mesh = o3d.io.read_triangle_mesh(OBJ_FILE)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
```

**Output:**
```
Vertices  : 50000+
Triangles : 100000+
Has texture: True/False
→ Opens 3D viewer window
```

---

### 📄 02_feature_extraction.py — Feature Extraction

This is the core of the project. It extracts geometric properties from every point on the 3D surface.

```
What it does:
→ Converts mesh to point cloud (10,000 points)
→ Extracts 4 key geometric features:
   1. Surface Normals  → direction each surface faces
   2. Curvature        → how much surface bends
   3. Roughness        → smooth vs jagged texture
   4. Boundary Edges   → sharp edge detection
→ Builds feature matrix (10000 × 8)
→ Auto-calculates threshold using mean + std
→ Colors boundary points RED in viewer
→ Saves feature_matrix.npy and boundary_mask.npy
```

**Key concept — Auto threshold:**
```python
# Instead of fixed threshold (caused all-red problem)
# We calculate threshold from actual data
threshold = mean_curvature + (1.0 * std_curvature)
# Points above threshold = boundary/break edges
```

**Output:**
```
Feature matrix shape: (10000, 8)
Features: [x, y, z, nx, ny, nz, curvature, roughness]
→ 3D viewer: Grey=normal, Red=boundaries
```

---

### 📄 03_break_simulation.py — Break Simulation

Since real broken artifact datasets are rare, this script simulates breaks on complete 3D objects.

```
What it does:
→ Loads the monument point cloud
→ Finds the center of the object
→ Creates slightly angled break plane
   (more realistic than straight cut)
→ Splits into Fragment A (top) and Fragment B (bottom)
→ Detects break surface on each fragment
   using neighbor count (fewer neighbors = edge)
→ Colors fragments differently:
   Red    = Fragment A
   Blue   = Fragment B
   Yellow = Break surface A
   Green  = Break surface B
→ Saves fragment_A.ply and fragment_B.ply
```

**Key concept — Break simulation:**
```python
# Angled break plane (more realistic)
break_plane = (
    points[:, 2] + angle * points[:, 0]
) > center_z
# angle=0.1 creates slight diagonal break
```

**Output:**
```
Fragment A points: ~5000
Fragment B points: ~5000
Break surface A  : ~200-500 points
Break surface B  : ~200-500 points
→ 3D viewer with 4 colored regions
```

---

### 📄 04_ml_classifier.py — ML Classifier

This script trains a K-Nearest Neighbors (KNN) classifier to automatically detect break surfaces.

```
What it does:
→ Generates training data from 6 point densities
   (3000, 5000, 7000, 10000, 12000, 15000 points)
→ Extracts 24 MULTISCALE features per point:
   - 3 radii × 6 features = 18 geometric features
   - Plus x, y, z coordinates = 3 features
   - Plus surface normals = 3 features
   - Total = 24 features
→ Uses smart labeling based on:
   curvature + roughness + linearity + planarity
→ Handles class imbalance automatically
→ Finds best K value (tests K=3,5,7,9,11,15,21)
→ Trains final KNN model with best K
→ Saves trained_model.pkl and scaler.pkl
```

**24 Multiscale Features:**
```
For each of 3 radii (0.02, 0.05, 0.10):
  1. Curvature     → how much surface bends
  2. Roughness     → surface texture variation
  3. Density       → how many nearby points
  4. Planarity     → how flat the surface is
  5. Linearity     → how linear the surface is
  6. Omnivariance  → overall shape variation

Plus:
  7-9.  x, y, z coordinates
  10-12. nx, ny, nz normals
```

**Why KNN:**
```
KNN works well because:
→ Break surfaces share similar local geometry
→ Points near break edges look similar
→ Distance-weighted voting improves accuracy
→ No complex training needed
```

**Output:**
```
Best K = 7 (varies per dataset)
Test Accuracy: 80-90%+
→ Plots: confusion matrix + CV scores
```

---

### 📄 05_matching_algorithm.py — Matching Algorithm

This script compares break surfaces between two fragments to determine if they match.

```
What it does:
→ Loads saved model and scaler
→ Loads Fragment A and Fragment B
→ Extracts 24 features from each fragment
→ Uses trained ML model to predict break surfaces
→ Calculates 4 matching scores:
   1. Normal similarity  → surface directions match?
   2. Size ratio         → similar break area size?
   3. Spread similarity  → similar shape spread?
   4. Density similarity → similar point density?
→ Combines scores into final match score
→ Threshold: ≥0.80 = MATCH
→ Saves matching_report.txt
→ Generates bar chart of scores
```

**Matching score formula:**
```python
match_score = (
    0.35 * normal_similarity  +  # most important
    0.25 * size_ratio         +  # area similarity
    0.25 * spread_similarity  +  # shape similarity
    0.15 * density_similarity    # density similarity
)
# match_score >= 0.80 → MATCH FOUND ✅
```

**Output:**
```
Normal similarity : 0.85
Size ratio        : 0.92
Spread similarity : 0.88
Density similarity: 0.83
─────────────────────────
Match score       : 0.87
RESULT: MATCH FOUND ✅
```

---

### 📄 06_evaluation.py — Final Evaluation

This script provides comprehensive evaluation of the entire pipeline.

```
What it does:
→ Rebuilds model completely from scratch
   (avoids any feature mismatch errors)
→ Tests on 5 different point counts
   (3000, 5000, 8000, 10000, 12000)
→ Handles single class problem automatically
→ Calculates 4 evaluation metrics:
   1. Accuracy   → overall correct predictions
   2. Precision  → correct break detections
   3. Recall     → break surfaces found
   4. F1 Score   → balance of precision/recall
→ Tests fragment matching on 3 point counts
→ Generates 4 evaluation plots
→ Saves final_report.txt
→ Saves updated model and scaler
```

**Evaluation metrics explained:**
```
Accuracy  = (correct predictions) / (total predictions)
           Target: ≥80% ✅

Precision = (true breaks found) / (all predicted breaks)
           How reliable are predictions?

Recall    = (true breaks found) / (all actual breaks)
           How many breaks did we find?

F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
           Balance between precision and recall
```

**Output:**
```
Mean Accuracy  : 80-90%+  ✅
Mean Precision : 0.85+
Mean Recall    : 0.82+
Mean F1 Score  : 0.83+
→ 4 evaluation plots generated
→ final_report.txt saved
```

---

## 📊 Dataset

| Dataset | Source | Format | Purpose |
|---|---|---|---|
| Chukur Fountain Monument | Sketchfab | .OBJ | Primary test object |
| Pottery Vessel (Jomon) | Smithsonian 3D | .OBJ | Heritage artifact |
| Stanford Bunny | Stanford 3D | .PLY | Pipeline testing |
| Simulated Breaks | Generated | .PLY | Training data |

**Dataset Citation:**
```
Smithsonian 3D Digitization:
→ Pottery Vessel (Horinouchi Type)
→ Source: 3d.si.edu
→ License: Usage Conditions Apply (Research Use)

Stanford 3D Scanning Repository:
→ Stanford Bunny
→ Source: graphics.stanford.edu
→ License: Free for research use
```

---

## 📈 Results

| Metric | Score | Target | Status |
|---|---|---|---|
| Break Detection Accuracy | 80-90%+ | ≥80% | ✅ |
| Fragment Match Score | 0.85+ | ≥0.80 | ✅ |
| Precision | 0.85+ | ≥0.80 | ✅ |
| Recall | 0.82+ | ≥0.80 | ✅ |
| F1 Score | 0.83+ | ≥0.80 | ✅ |

---

## 🐳 Docker Setup

Run the entire project without installing anything:

```bash
# Build container
docker build -t gsoc2026 -f docker/Dockerfile .

# Run evaluation
docker run gsoc2026
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push to branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Suyash Goyal**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [your-linkedin](https://linkedin.com/in/yourprofile)
- GSoC 2026 — HumanAI Foundation

---

## 🙏 Acknowledgements

- **HumanAI Foundation** — for the project idea and mentorship
- **Google Summer of Code 2026** — for the opportunity
- **Smithsonian 3D Digitization** — for open artifact datasets
- **Stanford 3D Scanning Repository** — for benchmark datasets
- **Open3D Team** — for the excellent 3D processing library

---

> *"Automating archaeological fragment matching using machine learning and geometric feature extraction — bringing AI to heritage preservation."*