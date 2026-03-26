# 01_load_visualize.py
import open3d as o3d
import numpy as np
import os

print("Libraries loaded")

# SET YOUR FILE PATH HERE
OBJ_FILE = r"C:\Users\Suyash goyal\Downloads\pottery-jug\source\HCM256\HCM256.obj"

# CHECK ALL FILES EXIST
folder = os.path.dirname(OBJ_FILE)
print("\nChecking files in folder:")
print("="*40)
for file in os.listdir(folder):
    print("Found:", file)
print("="*40)

# LOAD MONUMENT
print("\nLoading monument...")
mesh = o3d.io.read_triangle_mesh(
    OBJ_FILE,
    enable_post_processing=True
)

# BASIC INFORMATION
print("="*40)
print("MONUMENT INFO")
print("="*40)
print("Vertices  :", len(mesh.vertices))
print("Triangles :", len(mesh.triangles))
print("Has texture:", mesh.has_textures())
print("Has UVs    :", mesh.has_triangle_uvs())
print("="*40)

# VISUALIZE
mesh.compute_vertex_normals()

if mesh.has_textures():
    print("\nShowing WITH texture")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Monument WITH Texture",
        width=800,
        height=600
    )
else:
    print("\n No texture found")
    print("Showing WITHOUT texture...")
    # Paint grey color instead
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Monument WITHOUT Texture",
        width=800,
        height=600
    )

print("\nDone")
