# 04_ml_classifier.py
# Step 4: Train ML Classifier for Break Surface Detection

import open3d as o3d
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

print("="*40)
print("ML CLASSIFIER STARTED")
print("="*40)

# LOAD SAVED FEATURES
print("\nLoading saved features...")

feature_matrix = np.load(
    "results/feature_matrix.npy"
)
boundary_mask = np.load(
    "results/boundary_mask.npy"
)

print("Feature matrix shape:", feature_matrix.shape)
print("Total points        :", len(boundary_mask))
print("Break points        :", np.sum(boundary_mask))
print("Normal points       :", np.sum(~boundary_mask))
print("Features loaded")

# PREPARE TRAINING DATA
print("\nPreparing training data...")

# X = features, y = labels
X = feature_matrix
y = boundary_mask.astype(int)
# 1 = break surface
# 0 = normal surface

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution:")
print("  Normal surface (0):", np.sum(y == 0))
print("  Break surface  (1):", np.sum(y == 1))

# SCALE FEATURES
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled")

# SPLIT INTO TRAIN AND TEST
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      # 80% train 20% test
    random_state=42,
    stratify=y          # keep class balance
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print("Data split")

# TRAIN RANDOM FOREST MODEL
print("\nTraining Random Forest model...")
print("Please wait...")

model = RandomForestClassifier(
    n_estimators=100,   # 100 trees
    max_depth=10,       # tree depth
    random_state=42,
    n_jobs=-1           # use all CPU cores
)

model.fit(X_train, y_train)
print("Model trained")

# EVALUATE MODEL
print("\nEvaluating model...")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print("MODEL RESULTS")
print("="*40)
print("Accuracy:", round(accuracy * 100, 2), "%")

if accuracy >= 0.80:
    print("Target achieved  (≥80% accuracy)")
else:
    print(" Below 80% target — need improvement")

print("\nDetailed Report:")
print("="*40)
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Break"]
))

# FEATURE IMPORTANCE
print("\nFeature Importance:")
print("="*40)

feature_names = [
    "x", "y", "z",
    "normal_x", "normal_y", "normal_z",
    "curvature", "roughness"
]

importances = model.feature_importances_

for name, importance in zip(
    feature_names, importances
):
    bar = "█" * int(importance * 50)
    print(f"{name:12} {bar} {round(importance, 4)}")

# PLOT RESULTS
print("\nGenerating plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Feature Importance
axes[0].barh(
    feature_names,
    importances,
    color="steelblue"
)
axes[0].set_title("Feature Importance")
axes[0].set_xlabel("Importance Score")

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_title("Confusion Matrix")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Normal", "Break"])
axes[1].set_yticklabels(["Normal", "Break"])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# Add numbers to confusion matrix
for i in range(2):
    for j in range(2):
        axes[1].text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="black", fontsize=14
        )

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/ml_results.png")
plt.show()
print("Plot saved to results/ml_results.png")

# SAVE MODEL
print("\nSaving model...")

with open("results/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("results/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model saved")
print("Scaler saved")

# VISUALIZE PREDICTIONS
print("\nVisualizing predictions...")

# Load original point cloud
OBJ_FILE = r"C:\Users\Suyash goyal\Desktop\monument\untitledTexturePainted.obj"
mesh = o3d.io.read_triangle_mesh(OBJ_FILE)
pcd  = mesh.sample_points_uniformly(
    number_of_points=10000
)

# Get predictions for all points
all_predictions = model.predict(X_scaled)

points = np.asarray(pcd.points)

# Predicted normal = grey
normal_pcd = o3d.geometry.PointCloud()
normal_pcd.points = o3d.utility.Vector3dVector(
    points[all_predictions == 0]
)
normal_pcd.paint_uniform_color([0.7, 0.7, 0.7])

# Predicted break = red
break_pcd = o3d.geometry.PointCloud()
break_pcd.points = o3d.utility.Vector3dVector(
    points[all_predictions == 1]
)
break_pcd.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries(
    [normal_pcd, break_pcd],
    window_name="ML Predictions - Red=Break Surface",
    width=800,
    height=600
)

print("\n" + "="*40)
print("ML CLASSIFIER COMPLETE")
print("="*40)
print("\nFiles saved:")
print("→ results/trained_model.pkl")
print("→ results/scaler.pkl")
print("→ results/ml_results.png")
print("\nNext step: 05_matching_algorithm.py")
