# 02_feature_extraction.py
# Step 2: Extract Geometric Features from Monument

import open3d as o3d
import numpy as np
import os

print("="*40)
print("FEATURE EXTRACTION STARTED")
print("="*40)

# LOAD MONUMENT
OBJ_FILE = r"C:\Users\Suyash goyal\Downloads\pottery-jug\source\HCM256\HCM256.obj"

print("\nLoading monument...")
mesh = o3d.io.read_triangle_mesh(OBJ_FILE)
mesh.compute_vertex_normals()
print("Monument loaded")

# CONVERT TO POINT CLOUD
print("\nConverting to point cloud...")
pcd = mesh.sample_points_uniformly(
    number_of_points=10000
)
print("Point cloud created")
print("Total points:", len(pcd.points))

# FEATURE 1: SURFACE NORMALS
print("\nExtracting surface normals...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1,
        max_nn=30
    )
)
normals = np.asarray(pcd.normals)
print("Normals extracted")
print("Normal shape:", normals.shape)

# FEATURE 2: CURVATURE
print("\nCalculating curvature...")

def calculate_curvature(pcd, radius=0.05):
    curvature = []
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(
            pcd.points[i], radius
        )
        if k > 3:
            neighbor_normals = normals[idx, :]
            curvature.append(
                np.var(neighbor_normals)
            )
        else:
            curvature.append(0)

    return np.array(curvature)

curvature = calculate_curvature(pcd)
print("Curvature calculated")
print("Max curvature:", round(curvature.max(), 4))
print("Min curvature:", round(curvature.min(), 4))
print("Avg curvature:", round(curvature.mean(), 4))

# FEATURE 3: ROUGHNESS
print("\nCalculating roughness...")

def calculate_roughness(pcd, radius=0.05):
    roughness = []
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)

    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(
            pcd.points[i], radius
        )
        if k > 3:
            neighbors = points[idx, :]
            roughness.append(np.var(neighbors))
        else:
            roughness.append(0)

    return np.array(roughness)

roughness = calculate_roughness(pcd)
print("Roughness calculated")
print("Max roughness:", round(roughness.max(), 4))
print("Min roughness:", round(roughness.min(), 4))
print("Avg roughness:", round(roughness.mean(), 4))

# FEATURE 4: BOUNDARY EDGES
print("\nDetecting boundary edges...")

def detect_boundaries(pcd, curvature):

    # Auto calculate threshold
    mean_curv = np.mean(curvature)
    std_curv  = np.std(curvature)

    # Boundary = points above mean + 1 std
    threshold = mean_curv + (1.0 * std_curv)

    print("\nAuto threshold calculated:")
    print("Mean curvature :", round(mean_curv, 4))
    print("Std curvature  :", round(std_curv,  4))
    print("Threshold used :", round(threshold, 4))

    boundary_mask   = curvature > threshold
    boundary_points = np.asarray(
        pcd.points)[boundary_mask]

    return boundary_points, boundary_mask

boundary_points, boundary_mask = detect_boundaries(
    pcd, curvature
)
print("Boundaries detected ")
print("Boundary points:", len(boundary_points))
print("Normal points  :", len(np.asarray(pcd.points))
                          - len(boundary_points))

# BUILD FEATURE MATRIX
print("\nBuilding feature matrix...")

points  = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

feature_matrix = np.column_stack([
    points,           # x, y, z        (3 features)
    normals,          # nx, ny, nz      (3 features)
    curvature,        # curvature value (1 feature)
    roughness         # roughness value (1 feature)
])

print("Feature matrix built")
print("="*40)
print("FEATURE MATRIX SUMMARY")
print("="*40)
print("Shape          :", feature_matrix.shape)
print("Total points   :", feature_matrix.shape[0])
print("Total features :", feature_matrix.shape[1])
print("Features       :")
print("  [x, y, z, nx, ny, nz, curvature, roughness]")
print("="*40)

# SAVE FEATURES
print("\nSaving features...")
os.makedirs("results", exist_ok=True)
np.save("results/feature_matrix.npy", feature_matrix)
np.save("results/boundary_mask.npy",  boundary_mask)
print("Features saved to results/ folder")

# VISUALIZE BOUNDARY POINTS
print("\nVisualizing results...")
print("Red points  = boundary/break edges")
print("Grey points = normal surface")

# Normal points in grey
normal_pcd = o3d.geometry.PointCloud()
normal_pcd.points = o3d.utility.Vector3dVector(
    points[~boundary_mask]
)
normal_pcd.paint_uniform_color([0.7, 0.7, 0.7])

# Boundary points in red
boundary_pcd = o3d.geometry.PointCloud()
boundary_pcd.points = o3d.utility.Vector3dVector(
    boundary_points
)
boundary_pcd.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries(
    [normal_pcd, boundary_pcd],
    window_name="Feature Extraction - Red=Boundaries",
    width=800,
    height=600
)

print("\n" + "="*40)
print("FEATURE EXTRACTION COMPLETE")
print("="*40)
print("Next step: 03_break_simulation.py")
