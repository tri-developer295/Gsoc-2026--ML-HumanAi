# 03_break_simulation.py
# Step 3: Simulate Break on Monument

import open3d as o3d
import numpy as np
import os

print("="*40)
print("BREAK SIMULATION STARTED")
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
points = np.asarray(pcd.points)
print("Total points:", len(points))
print("Point cloud created")

# SIMULATE BREAK
print("\nSimulating break...")

def simulate_break(pcd):
    points = np.asarray(pcd.points)

    # Find center of object
    center_z = np.mean(points[:, 2])
    center_x = np.mean(points[:, 0])

    print("\nObject center:")
    print("Center X:", round(center_x, 4))
    print("Center Z:", round(center_z, 4))

    # Create slightly angled break plane
    # for realistic break simulation
    angle = 0.1  # small angle for realism
    break_plane = (
        points[:, 2] + angle * points[:, 0]
    ) > (center_z + angle * center_x)

    # Split into two fragments
    fragment_A_points = points[break_plane]
    fragment_B_points = points[~break_plane]

    # Create Fragment A point cloud
    fragment_A = o3d.geometry.PointCloud()
    fragment_A.points = o3d.utility.Vector3dVector(
        fragment_A_points
    )

    # Create Fragment B point cloud
    fragment_B = o3d.geometry.PointCloud()
    fragment_B.points = o3d.utility.Vector3dVector(
        fragment_B_points
    )

    return fragment_A, fragment_B

fragment_A, fragment_B = simulate_break(pcd)

print("\nBreak simulation complete")
print("Fragment A points:", len(fragment_A.points))
print("Fragment B points:", len(fragment_B.points))

# DETECT BREAK SURFACE
print("\nDetecting break surfaces...")

def get_break_surface(fragment, radius=0.05):

    # Estimate normals
    fragment.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )

    points  = np.asarray(fragment.points)
    normals = np.asarray(fragment.normals)

    # Find boundary points of fragment
    pcd_tree = o3d.geometry.KDTreeFlann(fragment)
    break_mask = []

    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(
            fragment.points[i], radius
        )
        # Break surface = fewer neighbors
        # (edge of fragment)
        if k < 8:
            break_mask.append(True)
        else:
            break_mask.append(False)

    break_mask = np.array(break_mask)
    break_points = points[break_mask]

    return break_points, break_mask

break_A, mask_A = get_break_surface(fragment_A)
break_B, mask_B = get_break_surface(fragment_B)

print("Break surface A points:", len(break_A))
print("Break surface B points:", len(break_B))
print("Break surfaces detected")

# SAVE FRAGMENTS
print("\nSaving fragments...")
os.makedirs("results", exist_ok=True)

# Save fragment point clouds
o3d.io.write_point_cloud(
    "results/fragment_A.ply", fragment_A
)
o3d.io.write_point_cloud(
    "results/fragment_B.ply", fragment_B
)

# Save break surface masks
np.save("results/break_mask_A.npy", mask_A)
np.save("results/break_mask_B.npy", mask_B)

print("Fragments saved to results/ folder")

# VISUALIZE FRAGMENTS
print("\nVisualizing fragments...")
print("Red    = Fragment A (top)")
print("Blue   = Fragment B (bottom)")
print("Yellow = Break surface A")
print("Green  = Break surface B")

# Color Fragment A red
fragment_A.paint_uniform_color([1, 0, 0])

# Color Fragment B blue
fragment_B.paint_uniform_color([0, 0, 1])

# Color break surface A yellow
break_pcd_A = o3d.geometry.PointCloud()
break_pcd_A.points = o3d.utility.Vector3dVector(
    break_A
)
break_pcd_A.paint_uniform_color([1, 1, 0])

# Color break surface B green
break_pcd_B = o3d.geometry.PointCloud()
break_pcd_B.points = o3d.utility.Vector3dVector(
    break_B
)
break_pcd_B.paint_uniform_color([0, 1, 0])

# Show all together
o3d.visualization.draw_geometries(
    [fragment_A, fragment_B,
     break_pcd_A, break_pcd_B],
    window_name="Break Simulation - Fragments",
    width=800,
    height=600
)

print("\n" + "="*40)
print("BREAK SIMULATION COMPLETE")
print("="*40)
print("\nResults saved:")
print("→ results/fragment_A.ply")
print("→ results/fragment_B.ply")
print("→ results/break_mask_A.npy")
print("→ results/break_mask_B.npy")
print("\nNext step: 04_ml_classifier.py")

