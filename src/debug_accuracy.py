# debug_accuracy.py
# Diagnose why accuracy is low

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

print("="*40)
print("DIAGNOSING LOW ACCURACY")
print("="*40)

# ─────────────────────────────────
# LOAD MONUMENT
# ─────────────────────────────────
OBJ_FILE = r"C:\Users\Suyash goyal\Downloads\pottery-jug\source\HCM256\HCM256.obj"

mesh = o3d.io.read_triangle_mesh(OBJ_FILE)
mesh.compute_vertex_normals()
pcd  = mesh.sample_points_uniformly(
    number_of_points=10000
)

points  = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

print(f"\nTotal points : {len(points)}")
print(f"X range      : {points[:,0].min():.3f}"
      f" to {points[:,0].max():.3f}")
print(f"Y range      : {points[:,1].min():.3f}"
      f" to {points[:,1].max():.3f}")
print(f"Z range      : {points[:,2].min():.3f}"
      f" to {points[:,2].max():.3f}")

# ─────────────────────────────────
# CHECK CLASS DISTRIBUTION
# ─────────────────────────────────
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30
    )
)

pcd_tree  = o3d.geometry.KDTreeFlann(pcd)
curvature = []

for i in range(len(points)):
    [k, idx, _] = pcd_tree.search_radius_vector_3d(
        pcd.points[i], 0.05
    )
    if k > 3:
        curvature.append(
            np.var(np.asarray(pcd.normals)[idx,:])
        )
    else:
        curvature.append(0)

curvature = np.array(curvature)

print(f"\nCurvature Stats:")
print(f"Min : {curvature.min():.6f}")
print(f"Max : {curvature.max():.6f}")
print(f"Mean: {curvature.mean():.6f}")
print(f"Std : {curvature.std():.6f}")

# ─────────────────────────────────
# CHECK BREAK LABEL DISTRIBUTION
# ─────────────────────────────────
def simulate_break(pcd, ax=0.0, ay=0.0):
    points   = np.asarray(pcd.points)
    center_x = np.mean(points[:,0])
    center_y = np.mean(points[:,1])
    center_z = np.mean(points[:,2])
    return (
        points[:,2] +
        ax * points[:,0] +
        ay * points[:,1]
    ) > (
        center_z +
        ax * center_x +
        ay * center_y
    )

def get_labels(pcd, mask, radius=0.05):
    points   = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    labels   = []
    for i in range(len(points)):
        [k, idx, _] = \
            pcd_tree.search_radius_vector_3d(
                pcd.points[i], radius
            )
        if k > 3:
            nb = mask[idx]
            if np.any(nb) and np.any(~nb):
                labels.append(1)
            else:
                labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)

mask   = simulate_break(pcd)
labels = get_labels(pcd, mask)

break_count  = np.sum(labels == 1)
normal_count = np.sum(labels == 0)
total        = len(labels)

print(f"\nLabel Distribution:")
print(f"Break points  : {break_count} "
      f"({round(break_count/total*100,1)}%)")
print(f"Normal points : {normal_count} "
      f"({round(normal_count/total*100,1)}%)")

if break_count/total < 0.05:
    print("\n⚠️  PROBLEM FOUND!")
    print("Too few break points!")
    print("Try larger radius in get_labels()")
elif break_count/total > 0.45:
    print("\n⚠️  PROBLEM FOUND!")
    print("Too many break points!")
    print("Break surface too wide!")
else:
    print("\n✅ Class distribution looks good!")

# ─────────────────────────────────
# PLOT DIAGNOSIS
# ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15,4))

# Plot 1: Curvature distribution
axes[0].hist(curvature, bins=50, color="steelblue")
axes[0].set_title("Curvature Distribution")
axes[0].set_xlabel("Curvature Value")
axes[0].set_ylabel("Count")
axes[0].axvline(
    x=curvature.mean(),
    color="red", linestyle="--",
    label="mean"
)
axes[0].legend()

# Plot 2: Class balance
axes[1].bar(
    ["Normal", "Break"],
    [normal_count, break_count],
    color=["steelblue", "red"]
)
axes[1].set_title("Class Distribution")
axes[1].set_ylabel("Count")
for i, v in enumerate([normal_count, break_count]):
    axes[1].text(
        i, v + 10,
        str(v),
        ha="center"
    )

# Plot 3: Point cloud colored by label
colors        = np.zeros((len(points), 3))
colors[labels == 0] = [0.7, 0.7, 0.7]  # grey
colors[labels == 1] = [1.0, 0.0, 0.0]  # red

axes[2].scatter(
    points[:,0],
    points[:,2],
    c=colors,
    s=0.5
)
axes[2].set_title("Break Points (Red) Side View")
axes[2].set_xlabel("X")
axes[2].set_ylabel("Z")

plt.tight_layout()
plt.savefig("results/diagnosis.png")
plt.show()

print("\n" + "="*40)
print("DIAGNOSIS COMPLETE")
print("="*40)
print("Check diagnosis.png in results/")
print("Share the output numbers above!")
