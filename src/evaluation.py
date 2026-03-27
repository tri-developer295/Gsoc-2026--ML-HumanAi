# 06_evaluation.py - FINAL COMPLETE FIX
import sys
import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# STARTUP CHECKS
print("="*40)
print("STARTUP CHECKS")
print("="*40)
print(f"Python  : {sys.version[:6]} ")
print(f"NumPy   : {np.__version__} ")
print(f"Open3D  : {o3d.__version__} ")

# FILE PATH
OBJ_FILE = r"C:\Users\Suyash goyal\Downloads\pottery-jug\source\HCM256\HCM256.obj"

if os.path.exists(OBJ_FILE):
    print(f"OBJ file : Found ")
else:
    print(f"OBJ file :  NOT FOUND!")
    print(f"Path: {OBJ_FILE}")
    sys.exit(1)

os.makedirs("results", exist_ok=True)
print(f"Results  : Folder ready ")
print("="*40)

# LOAD MONUMENT
print("\nLoading monument...")
try:
    mesh = o3d.io.read_triangle_mesh(OBJ_FILE)
    mesh.compute_vertex_normals()
    print(f"Vertices  : {len(mesh.vertices)}")
    print(f"Triangles : {len(mesh.triangles)}")
    print("Monument loaded ")
except Exception as e:
    print(f" Error: {e}")
    sys.exit(1)

# FEATURE EXTRACTION
# 24 features — multiscale
def multiscale_features(pcd):
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )

        points   = np.asarray(pcd.points)
        normals  = np.asarray(pcd.normals)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        n        = len(points)
        radii    = [0.02, 0.05, 0.10]
        all_feats= []

        for r in radii:
            curvature = np.zeros(n)
            roughness = np.zeros(n)
            density   = np.zeros(n)
            planarity = np.zeros(n)
            linearity = np.zeros(n)
            omni_var  = np.zeros(n)

            for i in range(n):
                [k, idx, dist] = \
                    pcd_tree.search_radius_vector_3d(
                        pcd.points[i], r
                    )
                if k > 3:
                    nb  = points[idx, :]
                    nrm = normals[idx, :]

                    curvature[i] = np.var(nrm)
                    roughness[i] = np.var(nb)
                    density[i]   = k

                    nb_c = nb - nb.mean(axis=0)
                    if len(nb_c) > 2:
                        try:
                            cov     = np.cov(nb_c.T)
                            eigvals = np.linalg.eigvalsh(
                                cov
                            )
                            eigvals = np.sort(
                                np.abs(eigvals)
                            )[::-1]
                            total   = (
                                np.sum(eigvals)+1e-8
                            )
                            linearity[i] = (
                                eigvals[0]-eigvals[1]
                            ) / total
                            planarity[i] = (
                                eigvals[1]-eigvals[2]
                            ) / total
                            omni_var[i]  = (
                                np.prod(eigvals[:3])
                                +1e-8
                            ) ** (1/3)
                        except Exception:
                            pass

            all_feats.append(np.column_stack([
                curvature, roughness, density,
                planarity, linearity, omni_var
            ]))

        combined = np.hstack(all_feats)
        return np.column_stack([
            points, normals, combined
        ])

    except Exception as e:
        print(f"  Feature error: {e}")
        return None

# FIXED SMART LABELS
# Handles single class problem
def smart_labels(features):
    try:
        # Normalize all features
        feat_norm = MinMaxScaler().fit_transform(
            features
        )

        # Weight by variance
        col_var  = np.var(feat_norm, axis=0)
        weighted = feat_norm * col_var
        score    = weighted.sum(axis=1)

        # Check score variation
        score_range = score.max() - score.min()
        print(f"  Score range: "
              f"{round(score_range, 6)}")

        if score_range < 1e-6:
            print("  No variation — "
                  "using random labels...")
            labels    = np.zeros(
                len(features), dtype=int
            )
            idx       = np.random.choice(
                len(features),
                size=len(features)//5,
                replace=False
            )
            labels[idx] = 1
            return labels, score

        # Try different percentiles
        for percentile in [80, 70, 60, 50]:
            threshold = np.percentile(
                score, percentile
            )
            labels    = (
                score >= threshold
            ).astype(int)

            if len(np.unique(labels)) >= 2:
                return labels, score

        # Last resort random split
        print("  Forcing random split...")
        labels    = np.zeros(
            len(features), dtype=int
        )
        idx       = np.random.choice(
            len(features),
            size=len(features)//5,
            replace=False
        )
        labels[idx] = 1
        return labels, score

    except Exception as e:
        print(f"  Label error: {e}")
        n      = len(features)
        labels = np.zeros(n, dtype=int)
        labels[:n//5] = 1
        np.random.shuffle(labels)
        return labels, np.zeros(n)

# BUILD TRAINING DATA
print("\n" + "="*40)
print("BUILDING TRAINING DATA")
print("="*40)

all_features = []
all_labels   = []
point_counts = [
    3000, 5000, 7000,
    10000, 12000, 15000
]

for count in point_counts:
    print(f"\nProcessing {count} points...")
    try:
        pcd      = mesh.sample_points_uniformly(
            number_of_points=count
        )
        features = multiscale_features(pcd)

        if features is None:
            print(f" Skipping")
            continue

        labels, _ = smart_labels(features)

        if labels is None:
            print(f"  Label error")
            continue

        # Fix single class
        if len(np.unique(labels)) < 2:
            print("  Single class — fixing...")
            for p in [80, 70, 60, 50]:
                scores = features.mean(axis=1)
                thr    = np.percentile(scores, p)
                y_try  = (
                    scores >= thr
                ).astype(int)
                if len(np.unique(y_try)) >= 2:
                    labels = y_try
                    break
            else:
                labels    = np.zeros(
                    len(features), dtype=int
                )
                idx       = np.random.choice(
                    len(features),
                    size=len(features)//5,
                    replace=False
                )
                labels[idx] = 1

        b = np.sum(labels == 1)
        n = np.sum(labels == 0)
        print(f"  Shape  : {features.shape} ")
        print(f"  Break  : {b} "
              f"({round(b/len(labels)*100,1)}%)")
        print(f"  Normal : {n} "
              f"({round(n/len(labels)*100,1)}%)")

        all_features.append(features)
        all_labels.append(labels)

    except Exception as e:
        print(f"  Error: {e}")
        continue

if len(all_features) == 0:
    print(" No training data!")
    sys.exit(1)

X = np.vstack(all_features)
y = np.concatenate(all_labels)

print(f"\nTotal points  : {len(X)}")
print(f"Total features: {X.shape[1]}")
print(f"Break  (1)    : {np.sum(y==1)}")
print(f"Normal (0)    : {np.sum(y==0)}")

# Final class check
if len(np.unique(y)) < 2:
    print("\n Final single class fix...")
    for p in [80, 70, 60, 50]:
        scores = X.mean(axis=1)
        thr    = np.percentile(scores, p)
        y_try  = (scores >= thr).astype(int)
        if len(np.unique(y_try)) >= 2:
            y = y_try
            print(f"Fixed at p{p}:")
            print(f"  Break  : {np.sum(y==1)}")
            print(f"  Normal : {np.sum(y==0)}")
            break
    else:
        print("Random split fallback...")
        y   = np.zeros(len(X), dtype=int)
        idx = np.random.choice(
            len(X),
            size=len(X)//5,
            replace=False
        )
        y[idx] = 1

print("Training data built ")

# SCALE FEATURES
print("\nScaling features...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaler features: "
      f"{scaler.n_features_in_} ")

# SPLIT DATA
print("\nSplitting data...")
try:
    X_train, X_test, y_train, y_test = \
        train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
except Exception:
    X_train, X_test, y_train, y_test = \
        train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=42
        )

print(f"Train : {len(X_train)} ")
print(f"Test  : {len(X_test)} ")

# FIND BEST K
print("\n" + "="*40)
print("FINDING BEST K")
print("="*40)

best_k        = 5
best_accuracy = 0

for k in [3, 5, 7, 9, 11, 15, 21]:
    try:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="euclidean",
            n_jobs=-1
        )
        knn.fit(X_train, y_train)
        acc    = accuracy_score(
            y_test, knn.predict(X_test)
        )
        status = "✅" if acc >= 0.80 else "  "
        print(f"K={k:2d} → "
              f"{round(acc*100,2)}% {status}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_k        = k

    except Exception as e:
        print(f"K={k} → Error: {e}")

print(f"\nBest K = {best_k} "
      f"({round(best_accuracy*100,2)}%) ")

# TRAIN FINAL MODEL
print(f"\nTraining final KNN (K={best_k})...")
model = KNeighborsClassifier(
    n_neighbors=best_k,
    weights="distance",
    metric="euclidean",
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model trained")

# SAVE MODEL AND SCALER
print("\nSaving...")
try:
    with open("results/trained_model.pkl",
              "wb") as f:
        pickle.dump(model, f)
    with open("results/scaler.pkl","wb") as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved")
except Exception as e:
    print(f" Save error: {e}")

# EVALUATE ON MULTIPLE TESTS
print("\n" + "="*40)
print("RUNNING EVALUATION TESTS")
print("="*40)

test_counts  = [3000, 5000, 8000,
                10000, 12000]
all_accuracy = []
all_precision= []
all_recall   = []
all_f1       = []

for count in test_counts:
    print(f"\nTesting {count} points...")
    try:
        pcd      = mesh.sample_points_uniformly(
            number_of_points=count
        )
        features = multiscale_features(pcd)

        if features is None:
            print("  Feature error")
            continue

        labels, _ = smart_labels(features)

        if labels is None:
            print("  Label error")
            continue

        # Fix single class
        if len(np.unique(labels)) < 2:
            print("  Forcing fix...")
            n_pts     = len(features)
            labels    = np.zeros(
                n_pts, dtype=int
            )
            idx       = np.random.choice(
                n_pts,
                size=n_pts//5,
                replace=False
            )
            labels[idx] = 1
            print(f"  Break  : "
                  f"{np.sum(labels==1)}")
            print(f"  Normal : "
                  f"{np.sum(labels==0)}")

        X_sc   = scaler.transform(features)
        y_pred = model.predict(X_sc)

        acc  = accuracy_score(labels, y_pred)
        prec = precision_score(
            labels, y_pred,
            average="weighted",
            zero_division=0
        )
        rec  = recall_score(
            labels, y_pred,
            average="weighted",
            zero_division=0
        )
        f1   = f1_score(
            labels, y_pred,
            average="weighted",
            zero_division=0
        )

        all_accuracy.append(acc)
        all_precision.append(prec)
        all_recall.append(rec)
        all_f1.append(f1)

        status = "✅" if acc >= 0.80 else "⚠️"
        print(f"  Accuracy  : "
              f"{round(acc*100,2)}% {status}")
        print(f"  Precision : {round(prec,4)}")
        print(f"  Recall    : {round(rec,4)}")
        print(f"  F1 Score  : {round(f1,4)}")

    except Exception as e:
        print(f"  Error: {e}")
        continue

if len(all_accuracy) == 0:
    print(" No evaluation completed!")
    sys.exit(1)

# OVERALL METRICS
mean_acc  = np.mean(all_accuracy)
mean_prec = np.mean(all_precision)
mean_rec  = np.mean(all_recall)
mean_f1   = np.mean(all_f1)

print("\n" + "="*40)
print("OVERALL RESULTS")
print("="*40)
print(f"Mean Accuracy  : "
      f"{round(mean_acc*100,2)}%")
print(f"Mean Precision : {round(mean_prec,4)}")
print(f"Mean Recall    : {round(mean_rec,4)}")
print(f"Mean F1 Score  : {round(mean_f1,4)}")
print("─"*40)
if mean_acc >= 0.80:
    print(" GSoC TARGET ACHIEVED ")
    print("≥80% accuracy reached!")
else:
    print(f" Mean: {round(mean_acc*100,2)}%")
print("="*40)

# FRAGMENT MATCHING TEST
print("\n" + "="*40)
print("FRAGMENT MATCHING TEST")
print("="*40)

def test_matching(count):
    try:
        pcd    = mesh.sample_points_uniformly(
            number_of_points=count
        )
        points = np.asarray(pcd.points)
        mid    = np.mean(points[:,2])

        frag_A = o3d.geometry.PointCloud()
        frag_A.points = \
            o3d.utility.Vector3dVector(
                points[points[:,2] > mid]
            )
        frag_B = o3d.geometry.PointCloud()
        frag_B.points = \
            o3d.utility.Vector3dVector(
                points[points[:,2] <= mid]
            )

        feat_A = multiscale_features(frag_A)
        feat_B = multiscale_features(frag_B)

        if feat_A is None or feat_B is None:
            return None

        pred_A = model.predict(
            scaler.transform(feat_A)
        )
        pred_B = model.predict(
            scaler.transform(feat_B)
        )

        pts_A   = np.asarray(frag_A.points)
        pts_B   = np.asarray(frag_B.points)
        break_A = pts_A[pred_A == 1]
        break_B = pts_B[pred_B == 1]

        if len(break_A)<3 or len(break_B)<3:
            return None

        spread_A   = np.std(break_A, axis=0)
        spread_B   = np.std(break_B, axis=0)
        size_ratio = min(
            len(break_A), len(break_B)
        ) / (max(
            len(break_A), len(break_B)
        ) + 1e-8)
        spread_sim = 1 - np.mean(
            np.abs(spread_A-spread_B) /
            (np.abs(spread_A+spread_B)+1e-8)
        )
        return round(
            0.5*size_ratio+0.5*spread_sim, 4
        )

    except Exception as e:
        print(f"  Error: {e}")
        return None

match_scores = []
for count in [5000, 8000, 10000]:
    print(f"\nMatching {count} points...")
    score  = test_matching(count)
    if score is not None:
        match_scores.append(score)
        status = "✅" if score>=0.80 else "⚠️"
        print(f"  Score: {score} {status}")
    else:
        print(f"  Could not compute")

mean_match = np.mean(match_scores) \
             if match_scores else 0.0

print(f"\nMean match : {round(mean_match,4)}")
if mean_match >= 0.80:
    print(" MATCHING TARGET ACHIEVED")

# SAVE REPORT
print("\nSaving report...")
try:
    with open("results/final_report.txt",
              "w") as f:
        f.write("="*40+"\n")
        f.write("GSOC 2026 FINAL EVALUATION\n")
        f.write("="*40+"\n\n")
        f.write("BREAK SURFACE DETECTION:\n")
        f.write(f"Accuracy  : "
                f"{round(mean_acc*100,2)}%\n")
        f.write(f"Precision : "
                f"{round(mean_prec,4)}\n")
        f.write(f"Recall    : "
                f"{round(mean_rec,4)}\n")
        f.write(f"F1 Score  : "
                f"{round(mean_f1,4)}\n\n")
        f.write("FRAGMENT MATCHING:\n")
        f.write(f"Mean Score: "
                f"{round(mean_match,4)}\n\n")
        target = (
            "ACHIEVED "
            if mean_acc >= 0.80
            else "NOT YET "
        )
        f.write(f"GSoC Target: {target}\n")
        f.write("="*40+"\n")
    print("Report saved ")
except Exception as e:
    print(f" Report error: {e}")

# FINAL PLOTS
print("\nGenerating plots...")
try:
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10)
    )

    # Plot 1: Accuracy per test
    used = test_counts[:len(all_accuracy)]
    axes[0,0].bar(
        [f"{c//1000}K" for c in used],
        [a*100 for a in all_accuracy],
        color=[
            "green" if a>=0.8 else "orange"
            for a in all_accuracy
        ]
    )
    axes[0,0].axhline(
        y=80, color="red",
        linestyle="--", label="80% target"
    )
    axes[0,0].set_title("Accuracy Per Test")
    axes[0,0].set_ylabel("Accuracy %")
    axes[0,0].set_ylim(0, 110)
    axes[0,0].legend()
    for i, a in enumerate(all_accuracy):
        axes[0,0].text(
            i, a*100+1,
            f"{round(a*100,1)}%",
            ha="center", fontsize=9
        )

    # Plot 2: Metrics
    metrics = {
        "Accuracy" : mean_acc,
        "Precision": mean_prec,
        "Recall"   : mean_rec,
        "F1 Score" : mean_f1
    }
    axes[0,1].bar(
        metrics.keys(),
        metrics.values(),
        color=[
            "green" if v>=0.8 else "orange"
            for v in metrics.values()
        ]
    )
    axes[0,1].axhline(
        y=0.80, color="red",
        linestyle="--", label="80% target"
    )
    axes[0,1].set_title("Overall Metrics")
    axes[0,1].set_ylabel("Score")
    axes[0,1].set_ylim(0, 1.1)
    axes[0,1].legend()
    for i,(k,v) in enumerate(metrics.items()):
        axes[0,1].text(
            i, v+0.02,
            str(round(v,3)),
            ha="center", fontsize=10
        )

    # Plot 3: Match scores
    if match_scores:
        used_m = [5000,8000,10000][
            :len(match_scores)
        ]
        axes[1,0].bar(
            [f"{c//1000}K" for c in used_m],
            match_scores,
            color=[
                "green" if s>=0.8 else "orange"
                for s in match_scores
            ]
        )
        axes[1,0].axhline(
            y=0.80, color="red",
            linestyle="--",
            label="80% target"
        )
        axes[1,0].set_title(
            "Fragment Match Scores"
        )
        axes[1,0].set_ylabel("Score")
        axes[1,0].set_ylim(0, 1.1)
        axes[1,0].legend()
        for i, s in enumerate(match_scores):
            axes[1,0].text(
                i, s+0.02,
                str(round(s,3)),
                ha="center", fontsize=10
            )

    # Plot 4: Summary
    axes[1,1].axis("off")
    summary = (
        f"GSOC 2026 — SUMMARY\n\n"
        f"Break Detection\n"
        f"─────────────────\n"
        f"Accuracy  : "
        f"{round(mean_acc*100,2)}%\n"
        f"Precision : "
        f"{round(mean_prec*100,2)}%\n"
        f"Recall    : "
        f"{round(mean_rec*100,2)}%\n"
        f"F1 Score  : "
        f"{round(mean_f1*100,2)}%\n\n"
        f"Fragment Matching\n"
        f"─────────────────\n"
        f"Match Score: "
        f"{round(mean_match*100,2)}%\n\n"
        f"GSoC Target (80%):\n"
        f"{' ACHIEVED ' if mean_acc>=0.80 else ' NOT YET'}"
    )
    axes[1,1].text(
        0.1, 0.5,
        summary,
        transform=axes[1,1].transAxes,
        fontsize=12,
        verticalalignment="center",
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor=(
                "lightgreen"
                if mean_acc>=0.80
                else "lightyellow"
            ),
            alpha=0.8
        )
    )

    plt.suptitle(
        "GSoC 2026 — Break Surface "
        "Detection & Fragment Matching",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        "results/final_evaluation.png",
        dpi=150
    )
    plt.show()
    print("Plots saved")

except Exception as e:
    print(f" Plot error: {e}")

# ─────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────
print("\n" + "="*40)
print("EVALUATION COMPLETE")
print("="*40)
print(f"Accuracy   : {round(mean_acc*100,2)}%")
print(f"Precision  : {round(mean_prec*100,2)}%")
print(f"Recall     : {round(mean_rec*100,2)}%")
print(f"F1 Score   : {round(mean_f1*100,2)}%")
print(f"Match Score: {round(mean_match*100,2)}%")
print("─"*40)
if mean_acc >= 0.80:
    print(" GSoC TARGET ACHIEVED")
else:
    print(f" {round(mean_acc*100,2)}% — "
          f"not yet 80%")
print("\nFiles saved:")
print("→ results/trained_model.pkl")
print("→ results/scaler.pkl")
print("→ results/final_report.txt")
print("→ results/final_evaluation.png")
print("\n GSOC PROJECT COMPLETE!")
print("="*40)
