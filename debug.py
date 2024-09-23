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
        if radius > 5:  # Assuming holes are usually smaller, adjust threshold if needed
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

# Main function
if __name__ == "__main__":
    file_path = r"C:\Users\Administrator\Desktop\test_step\钣金.STEP"
    shape = load_step_file(file_path)
    holes = count_holes(shape)
    print(f"Number of holes: {holes}")







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

