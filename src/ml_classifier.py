# 04_ml_classifier.py - FIXED VERSION
import open3d as o3d
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("="*40)
print("FIXED ML CLASSIFIER")
print("="*40)

# LOAD MONUMENT
OBJ_FILE = r"C:\Users\Suyash goyal\Downloads\pottery-jug\source\HCM256\HCM256.obj"

print("\nLoading monument...")
mesh = o3d.io.read_triangle_mesh(OBJ_FILE)
mesh.compute_vertex_normals()
print("Monument loaded")

# RICH FEATURE EXTRACTION
def extract_rich_features(pcd, radius=0.05):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )

    points   = np.asarray(pcd.points)
    normals  = np.asarray(pcd.normals)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    curvature  = []
    roughness  = []
    density    = []
    normal_var = []
    point_var  = []
    avg_dist   = []

    for i in range(len(points)):
        [k, idx, dist] = \
            pcd_tree.search_radius_vector_3d(
                pcd.points[i], radius
            )
        if k > 3:
            nb  = points[idx, :]
            nrm = normals[idx, :]
            curvature.append(np.var(nrm))
            roughness.append(np.var(nb))
            density.append(k)
            normal_var.append(
                np.mean(np.var(nrm, axis=0))
            )
            point_var.append(
                np.mean(np.var(nb, axis=0))
            )
            avg_dist.append(
                np.mean(np.sqrt(dist))
            )
        else:
            curvature.append(0)
            roughness.append(0)
            density.append(k)
            normal_var.append(0)
            point_var.append(0)
            avg_dist.append(0)

    curvature  = np.array(curvature)
    roughness  = np.array(roughness)
    density    = np.array(density)
    normal_var = np.array(normal_var)
    point_var  = np.array(point_var)
    avg_dist   = np.array(avg_dist)

    features = np.column_stack([
        points,
        normals,
        curvature,
        roughness,
        density,
        normal_var,
        point_var,
        avg_dist,
        curvature * roughness,
        normal_var / (avg_dist + 1e-8)
    ])

    return features, curvature, roughness

# FIXED LABELING STRATEGY
# Use curvature threshold instead
# of neighbor-side comparison
def get_smart_labels(curvature, roughness,
                     density):

    # Normalize each feature 0 to 1
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-8:
            return arr * 0
        return (arr - mn) / (mx - mn)

    curv_norm  = normalize(curvature)
    rough_norm = normalize(roughness)
    dens_norm  = normalize(density)

    # Break score = high curvature +
    # high roughness + low density
    break_score = (
        0.5 * curv_norm  +
        0.3 * rough_norm +
        0.2 * (1 - dens_norm)
    )

    # Use top 20% as break surface
    threshold = np.percentile(break_score, 80)
    labels    = (break_score >= threshold
                 ).astype(int)

    return labels, break_score

# GENERATE TRAINING DATA
# Multiple point densities
print("\nGenerating training data...")

all_features = []
all_labels   = []

point_counts = [5000, 8000, 10000,
                12000, 15000]

for count in point_counts:
    print(f"Processing {count} points...")

    pcd = mesh.sample_points_uniformly(
        number_of_points=count
    )

    features, curvature, roughness = \
        extract_rich_features(pcd)

    density = features[:, 8]

    labels, _ = get_smart_labels(
        curvature, roughness, density
    )

    all_features.append(features)
    all_labels.append(labels)

    print(f"  Break  : {np.sum(labels==1)}")
    print(f"  Normal : {np.sum(labels==0)}")

X = np.vstack(all_features)
y = np.concatenate(all_labels)

print(f"\nTotal points  : {len(X)}")
print(f"Break points  : {np.sum(y==1)}"
      f" ({round(np.sum(y==1)/len(y)*100,1)}%)")
print(f"Normal points : {np.sum(y==0)}"
      f" ({round(np.sum(y==0)/len(y)*100,1)}%)")
print("Training data generated")

# SCALE FEATURES
print("\nScaling features...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled")

# SPLIT DATA
X_train, X_test, y_train, y_test = \
    train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

print(f"\nTraining : {len(X_train)} samples")
print(f"Testing  : {len(X_test)} samples")

# TRAIN MODEL
print("\nTraining model...")
print("Please wait...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model trained")

# EVALUATE
print("\nEvaluating...")

y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

cv_scores = cross_val_score(
    model, X_scaled, y,
    cv=5, n_jobs=-1
)

print("\n" + "="*40)
print("RESULTS")
print("="*40)
print(f"Accuracy   : {round(accuracy*100, 2)}%")
print(f"CV Mean    : "
      f"{round(cv_scores.mean()*100, 2)}%")
print(f"CV Std     : "
      f"{round(cv_scores.std()*100, 2)}%")

if accuracy >= 0.80:
    print("\n🎉 TARGET ACHIEVED")
    print("≥80% accuracy reached!")
else:
    print(f"\n {round(accuracy*100,2)}% — "
          f"getting closer!")

print("\nDetailed Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Break"]
))

# FEATURE IMPORTANCE
feature_names = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "curvature", "roughness",
    "density", "normal_var",
    "point_var", "avg_dist",
    "curv×rough", "nvar/dist"
]

importances = model.feature_importances_

print("\nFeature Importance:")
print("="*40)
for name, imp in sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
):
    bar = "█" * int(imp * 50)
    print(f"{name:12} {bar} {round(imp,4)}")

# SAVE MODEL
print("\nSaving model...")
os.makedirs("results", exist_ok=True)

with open("results/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("results/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model saved")

# VISUALIZE PREDICTIONS
print("\nVisualizing predictions...")

pcd = mesh.sample_points_uniformly(
    number_of_points=10000
)
features_vis, curv_vis, rough_vis = \
    extract_rich_features(pcd)

X_vis  = scaler.transform(features_vis)
y_vis  = model.predict(X_vis)
pts    = np.asarray(pcd.points)

# Normal = grey
normal_pcd = o3d.geometry.PointCloud()
normal_pcd.points = o3d.utility.Vector3dVector(
    pts[y_vis == 0]
)
normal_pcd.paint_uniform_color([0.7, 0.7, 0.7])

# Break = red
break_pcd = o3d.geometry.PointCloud()
break_pcd.points = o3d.utility.Vector3dVector(
    pts[y_vis == 1]
)
break_pcd.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries(
    [normal_pcd, break_pcd],
    window_name="Fixed Predictions",
    width=800,
    height=600
)

# PLOT RESULTS
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
axes[0].imshow(cm, cmap="Blues")
axes[0].set_title(
    f"Confusion Matrix\n"
    f"Accuracy: {round(accuracy*100,2)}%"
)
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Normal", "Break"])
axes[0].set_yticklabels(["Normal", "Break"])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        axes[0].text(
            j, i, str(cm[i][j]),
            ha="center", fontsize=14
        )

# Feature Importance
sorted_idx = np.argsort(importances)
axes[1].barh(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color="steelblue"
)
axes[1].set_title("Feature Importance")
axes[1].set_xlabel("Score")

# CV Scores
axes[2].bar(
    [f"Fold {i+1}" for i in range(5)],
    cv_scores * 100,
    color=[
        "green" if s >= 0.8 else "orange"
        for s in cv_scores
    ]
)
axes[2].axhline(
    y=80, color="red",
    linestyle="--",
    label="80% target"
)
axes[2].set_title("Cross Validation")
axes[2].set_ylabel("Accuracy %")
axes[2].set_ylim(0, 110)
axes[2].legend()
for i, s in enumerate(cv_scores):
    axes[2].text(
        i, s*100+1,
        f"{round(s*100,1)}%",
        ha="center", fontsize=9
    )

plt.tight_layout()
plt.savefig("results/ml_results.png")
plt.show()

print("\n" + "="*40)
print("FIXED ML CLASSIFIER COMPLETE")
print("="*40)
print("Next: Run 05_matching_algorithm.py")