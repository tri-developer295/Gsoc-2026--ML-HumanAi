# 05_matching_algorithm.py
# Step 5: Match Break Surfaces Between Fragments

import open3d as o3d
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

print("="*40)
print("MATCHING ALGORITHM STARTED")
print("="*40)

# ─────────────────────────────────
# LOAD SAVED DATA
# ─────────────────────────────────
print("\nLoading saved data...")

with open("results/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("results/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

fragment_A = o3d.io.read_point_cloud(
    "results/fragment_A.ply"
)
fragment_B = o3d.io.read_point_cloud(
    "results/fragment_B.ply"
)

print("Model loaded ✅")
print("Fragments loaded ✅")
print("Fragment A points:", len(fragment_A.points))
print("Fragment B points:", len(fragment_B.points))

# ─────────────────────────────────
# UPDATED FEATURE EXTRACTION
# 14 features — matches Step 4
# ─────────────────────────────────
def extract_features(pcd, radius=0.05):

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

    # SAME 14 features as Step 4 ✅
    features = np.column_stack([
        points,                         # x,y,z     (3)
        normals,                        # nx,ny,nz   (3)
        curvature,                      # curvature  (1)
        roughness,                      # roughness  (1)
        density,                        # density    (1)
        normal_var,                     # normal var (1)
        point_var,                      # point var  (1)
        avg_dist,                       # avg dist   (1)
        curvature * roughness,          # interaction(1)
        normal_var / (avg_dist + 1e-8)  # ratio      (1)
    ])

    return features, curvature, roughness

# ─────────────────────────────────
# EXTRACT FEATURES FROM FRAGMENTS
# ─────────────────────────────────
print("\nExtracting features from fragments...")

print("Extracting Fragment A features...")
features_A, curv_A, rough_A = extract_features(
    fragment_A
)

print("Extracting Fragment B features...")
features_B, curv_B, rough_B = extract_features(
    fragment_B
)

print("Features extracted ✅")
print("Feature shape A:", features_A.shape)
print("Feature shape B:", features_B.shape)

# ─────────────────────────────────
# PREDICT BREAK SURFACES
# ─────────────────────────────────
print("\nPredicting break surfaces...")

features_A_scaled = scaler.transform(features_A)
features_B_scaled = scaler.transform(features_B)

pred_A = model.predict(features_A_scaled)
pred_B = model.predict(features_B_scaled)

points_A = np.asarray(fragment_A.points)
points_B = np.asarray(fragment_B.points)

break_A = points_A[pred_A == 1]
break_B = points_B[pred_B == 1]

print("Break surface A points:", len(break_A))
print("Break surface B points:", len(break_B))
print("Break surfaces predicted ✅")

# ─────────────────────────────────
# MATCH BREAK SURFACES
# ─────────────────────────────────
print("\nMatching break surfaces...")

def match_surfaces(break_A, break_B):

    if len(break_A) == 0 or len(break_B) == 0:
        print("❌ No break surfaces found!")
        return 0.0, False, {}

    # Center of each break surface
    center_A = np.mean(break_A, axis=0)
    center_B = np.mean(break_B, axis=0)

    # Spread of each break surface
    spread_A = np.std(break_A, axis=0)
    spread_B = np.std(break_B, axis=0)

    # Normal direction similarity
    normal_A      = break_A - center_A
    normal_B      = break_B - center_B
    mean_normal_A = np.mean(normal_A, axis=0)
    mean_normal_B = np.mean(normal_B, axis=0)

    norm_A = mean_normal_A / (
        np.linalg.norm(mean_normal_A) + 1e-8
    )
    norm_B = mean_normal_B / (
        np.linalg.norm(mean_normal_B) + 1e-8
    )

    # Score 1: Normal similarity
    normal_similarity = abs(np.dot(norm_A, norm_B))

    # Score 2: Size similarity
    size_ratio = min(
        len(break_A), len(break_B)
    ) / max(
        len(break_A), len(break_B)
    )

    # Score 3: Spread similarity
    spread_similarity = 1 - np.mean(
        np.abs(spread_A - spread_B) /
        (np.abs(spread_A + spread_B) + 1e-8)
    )

    # Score 4: Density similarity
    density_A = len(break_A) / (
        np.linalg.norm(spread_A) + 1e-8
    )
    density_B = len(break_B) / (
        np.linalg.norm(spread_B) + 1e-8
    )
    density_similarity = min(
        density_A, density_B
    ) / (max(density_A, density_B) + 1e-8)

    # Combined score
    match_score = (
        0.35 * normal_similarity  +
        0.25 * size_ratio         +
        0.25 * spread_similarity  +
        0.15 * density_similarity
    )

    is_match = match_score >= 0.80

    return match_score, is_match, {
        "normal_similarity"  : round(normal_similarity,  4),
        "size_ratio"         : round(size_ratio,          4),
        "spread_similarity"  : round(spread_similarity,   4),
        "density_similarity" : round(density_similarity,  4),
        "match_score"        : round(match_score,         4)
    }

match_score, is_match, details = match_surfaces(
    break_A, break_B
)

# ─────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────
print("\n" + "="*40)
print("MATCHING RESULTS")
print("="*40)
print("Normal similarity  :", details["normal_similarity"])
print("Size ratio         :", details["size_ratio"])
print("Spread similarity  :", details["spread_similarity"])
print("Density similarity :", details["density_similarity"])
print("─"*40)
print("Match score        :", details["match_score"])
print("Match threshold    : 0.80")
print("─"*40)

if is_match:
    print("RESULT: MATCH FOUND ✅")
    print("Fragments belong together!")
else:
    print("RESULT: NO MATCH ❌")
    print("Fragments do not match")

print("="*40)

# ─────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────
print("\nSaving results...")
os.makedirs("results", exist_ok=True)

with open("results/matching_report.txt", "w") as f:
    f.write("="*40 + "\n")
    f.write("MATCHING REPORT\n")
    f.write("="*40 + "\n")
    for key, value in details.items():
        f.write(f"{key}: {value}\n")
    f.write(f"is_match: {is_match}\n")

print("Report saved ✅")

# ─────────────────────────────────
# PLOT MATCHING SCORES
# ─────────────────────────────────
print("\nGenerating matching plot...")

labels = [
    "Normal\nSimilarity",
    "Size\nRatio",
    "Spread\nSimilarity",
    "Density\nSimilarity",
    "Final\nMatch Score"
]

values = [
    details["normal_similarity"],
    details["size_ratio"],
    details["spread_similarity"],
    details["density_similarity"],
    details["match_score"]
]

colors = [
    "steelblue",
    "steelblue",
    "steelblue",
    "steelblue",
    "green" if is_match else "red"
]

plt.figure(figsize=(12, 5))
bars = plt.bar(labels, values, color=colors)
plt.axhline(
    y=0.80,
    color="red",
    linestyle="--",
    label="80% threshold"
)
plt.ylim(0, 1.1)
plt.title("Fragment Matching Scores")
plt.ylabel("Score")
plt.legend()

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        str(round(value, 2)),
        ha="center",
        fontsize=12
    )

plt.tight_layout()
plt.savefig("results/matching_scores.png")
plt.show()
print("Plot saved ✅")

# ─────────────────────────────────
# VISUALIZE MATCHING
# ─────────────────────────────────
print("\nVisualizing matching result...")

points_A = np.asarray(fragment_A.points).copy()
points_B = np.asarray(fragment_B.points).copy()

# Fragment A → move up
frag_A_vis = o3d.geometry.PointCloud()
frag_A_vis.points = o3d.utility.Vector3dVector(
    points_A + [0, 0, 0.8]
)
frag_A_vis.paint_uniform_color([1, 0, 0])

# Fragment B → stays
frag_B_vis = o3d.geometry.PointCloud()
frag_B_vis.points = o3d.utility.Vector3dVector(
    points_B
)
frag_B_vis.paint_uniform_color([0, 0, 1])

# Break surface A → yellow
break_A_vis = o3d.geometry.PointCloud()
break_A_vis.points = o3d.utility.Vector3dVector(
    break_A + [0, 0, 0.8]
)
break_A_vis.paint_uniform_color([1, 1, 0])

# Break surface B → green
break_B_vis = o3d.geometry.PointCloud()
break_B_vis.points = o3d.utility.Vector3dVector(
    break_B
)
break_B_vis.paint_uniform_color([0, 1, 0])

o3d.visualization.draw_geometries(
    [frag_A_vis,  frag_B_vis,
     break_A_vis, break_B_vis],
    window_name="Fragment Matching Result",
    width=800,
    height=600
)

print("\n" + "="*40)
print("MATCHING ALGORITHM COMPLETE ✅")
print("="*40)
print("\nFiles saved:")
print("→ results/matching_report.txt")
print("→ results/matching_scores.png")
print("\nNext step: 06_evaluation.py")
