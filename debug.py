"""
step读取测试
"""
# **************************occwl****************************
# from occwl.io import load_step
# from occwl.graph import face_adjacency

# step_file = r"C:\Users\Administrator\Desktop\test_step\钣金.STEP"
# solid = load_step(step_file)[0]
# print(solid)
# graph = face_adjacency(solid)  # graph只有面和边的关系信息，没有属性信息
# print(graph)

"""
get the number of holes in a step file
(because the step file always split a complete hole into several curved faces, 
so the number this function returns may not be accurate)
"""
# ************************pythonocc**************************
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import topods_Face, topods_Edge
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties

# Step 1: Load the STEP file
def load_step_file(file_path):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(file_path)
    if status == 1:  # STEP file successfully loaded
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
        return shape
    else:
        raise Exception("Error: Could not load STEP file.")

# Step 2: Check if the face is an internal cylindrical hole
def is_hole(face):
    # Get the surface geometry of the face
    surface = BRepAdaptor_Surface(face)
    
    # Check if it's a cylindrical surface
    if surface.GetType() == GeomAbs_Cylinder:
        # Get the cylinder radius and orientation
        cylinder = surface.Cylinder()
        radius = cylinder.Radius()

        # Exclude large radii (for example, set a threshold based on part size)
        if radius > 10:  # Assuming holes are usually smaller, adjust threshold if needed
            return False
        
        # Check the face orientation to distinguish between holes and external surfaces
        if face.Orientation() == TopAbs_REVERSED:
            return True  # Holes are usually reversed in orientation (facing inward)
    
    return False

# Step 3: Traverse the faces and detect cylindrical holes
def count_holes(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    hole_count = 0

    while explorer.More():
        face = topods_Face(explorer.Current())
        if is_hole(face):
            hole_count += 1
        explorer.Next()

    return hole_count

# Step 2: Check if a face represents a bend (cylindrical surface with a small radius)
def is_bend_face(face, radius_threshold=10.0):
    # Get the surface geometry of the face
    surface = BRepAdaptor_Surface(face)
    
    if surface.GetType() == GeomAbs_Cylinder:
        # Get the cylinder radius
        cylinder = surface.Cylinder()
        radius = cylinder.Radius()
        
        # Use a radius threshold to identify small-radius bends (adjust threshold as needed)
        if radius < radius_threshold:
            return True
    return False

# Step 3: Count the number of bends in the shape
def count_bends(shape, radius_threshold=10.0):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    bend_count = 0

    while explorer.More():
        face = topods_Face(explorer.Current())
        
        if is_bend_face(face, radius_threshold):
            bend_count += 1
        
        explorer.Next()

    return bend_count


# Main function
if __name__ == "__main__":
    file_path = r"C:\Users\Administrator\Desktop\test_step\圆环.stp"
    shape = load_step_file(file_path)
    holes = count_holes(shape)
    bends = count_bends(shape)
    print(f"Number of holes: {holes}, Number of bends: {bends}")



"""
get the three-view drawing from a step file
"""
# from OCC.Core.STEPControl import STEPControl_Reader
# from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
# from OCC.Core.gp import gp_Pln, gp_Dir, gp_Ax3, gp_Pnt
# from OCC.Core.BRepTools import breptools_Write
# from OCC.Display.SimpleGui import init_display



# def read_step_file(file_path):
#     """Read a STEP file and return the loaded shape."""
#     step_reader = STEPControl_Reader()
#     status = step_reader.ReadFile(file_path)
#     if status != 1:
#         raise ValueError("Error reading the STEP file.")
#     step_reader.TransferRoots()
#     return step_reader.Shape()

# def project_shape_to_plane(shape, plane):
#     """Project the given shape onto the specified plane."""
#     # Projecting shape onto the plane
#     projection = BRepAlgoAPI_Section(shape, plane, False)
#     projection.Build()
#     if not projection.IsDone():
#         raise RuntimeError("Projection failed.")
#     return projection.Shape()

# def create_projection_planes():
#     """Create the planes for top, front, and side views."""
#     top_plane = gp_Pln(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)))  # Z-axis for top view
#     front_plane = gp_Pln(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))  # X-axis for front view
#     side_plane = gp_Pln(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))  # Y-axis for side view
#     return top_plane, front_plane, side_plane

# def export_projection_as_brep(projection_shape, output_path):
#     """Export the projected shape as a BREP file."""
#     breptools_Write(projection_shape, output_path)

# def save_view_as_image(display, filename):
#     """Save the current view of the display as an image."""
#     display.View.Dump(filename)
#     print(f"View saved as {filename}")

# def main(step_file_path):
#     # Initialize the 3D viewer
#     display, start_display, add_menu, add_function_to_menu = init_display()

#     # Read the STEP file
#     shape = read_step_file(step_file_path)

#     # Create planes for projections
#     top_plane, front_plane, side_plane = create_projection_planes()

#     # Project the shape onto each plane
#     top_view = project_shape_to_plane(shape, top_plane)
#     front_view = project_shape_to_plane(shape, front_plane)
#     side_view = project_shape_to_plane(shape, side_plane)

#     # Display the projections
#     display.DisplayShape(top_view, update=True)
#     save_view_as_image(display, "top_view.png")
    
#     display.DisplayShape(front_view, update=True)
#     save_view_as_image(display, "front_view.png")
    
#     display.DisplayShape(side_view, update=True)
#     save_view_as_image(display, "side_view.png")

#     print("All views have been saved as images.")

# # Replace 'your_step_file.step' with the path to your STEP file.
# main(r"C:\Users\Administrator\Desktop\test_step\圆环.stp")



"""
将测试集结果匹配到文件名,保存csv
"""
# import pandas

# # csv_path = r"E:\LGJ\program\UV-Net\results\regression\test_results_0924_143716_0.7874.csv" # v1
# # txt_path = r"E:\Project\AutoPricing\datasets\atwcad\v1\test.txt"
# csv_path = r"E:\LGJ\program\UV-Net\results\regression\test_results_1008_152433_0.7975.csv" 
# txt_path = r"E:\Project\AutoPricing\datasets\atwcad\test.txt"
# with open(txt_path, "r") as f:
#     lines = f.readlines()
# codes = [l.strip().split("  ")[0] for l in lines]

# df = pandas.read_csv(csv_path)
# df["code"] = codes
# df.to_csv(csv_path, index=False)




"""
切分数据集
"""
# raw_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\dataset.txt"
# train_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\train.txt"
# val_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\val.txt"
# test_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\test.txt"

# with open(raw_txt_path, "r") as f:
#     lines = f.readlines()


# import random
# random.shuffle(lines)

# train_num = int(len(lines) * 0.8)
# val_num = int(len(lines) * 0.1)
# test_num = len(lines) - train_num - val_num


# with open(train_txt_path, "w") as f:
#     f.writelines(lines[:train_num])

# with open(val_txt_path, "w") as f:
#     f.writelines(lines[train_num:train_num+val_num])

# with open(test_txt_path, "w") as f:
#     f.writelines(lines[train_num+val_num:])



# raw_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\v0_1\dataset.txt"
# dst_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\v0_1\test.txt"
# ref_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\test.txt"

# with open(raw_txt_path, "r") as f:
#     lines = f.readlines()

# with open(ref_txt_path, "r") as f:
#     ref_lines = f.readlines()

# ref = []
# for l in ref_lines:
#     code, _ = l.strip().split("  ")
#     ref.append(code)

# new_lines = []
# for l in lines:
#     code, _ = l.strip().split("  ")
#     if code in ref:
#         new_lines.append(l)

# with open(dst_txt_path, "w") as f:
#     f.writelines(new_lines)


