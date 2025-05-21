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
get the three-view drawing from a step file(not accurate)
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
# csv_path = r"E:\LGJ\program\UV-Net\results\regression\test_results_1108_162611_0.7926.csv" 
# txt_path = r"E:\Project\AutoPricing\datasets\atwcad\test.txt"
# with open(txt_path, "r") as f:
#     lines = f.readlines()
# codes = []
# nameCodes = []
# for l in lines:
#     code, annostr = l.strip().rsplit("  ", 1)
#     values = [float(x) for x in annostr.split(' ')]
#     nameCode = values[0]
#     codes.append(code)
#     nameCodes.append(nameCode)

# df = pandas.read_csv(csv_path)
# df["code"] = codes
# df["nameCode"] = nameCodes
# df.to_csv(csv_path, index=False)




"""
切分数据集
"""
# raw_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\焊接件.txt"
# train_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\train.txt"
# val_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\val.txt"

# with open(raw_txt_path, "r") as f:
#     lines = f.readlines()

# import random
# random.shuffle(lines)

# train_num = int(len(lines) * 0.8)

# with open(train_txt_path, "w") as f:
#     f.writelines(lines[:train_num])

# with open(val_txt_path, "w") as f:
#     f.writelines(lines[train_num:])

src = r"E:\Project\AutoPricing\datasets\atwcad\train.txt"

with open(src, "r") as f:
    lines = f.readlines()

new_lines = []
for l in lines:
    code, varstr = l.strip().split("  ")
    values = varstr.split(' ')
    vars, label = values[:-2], values[-1]
    new_line = code + "  " + " ".join(vars) + " " + label + "\n"
    new_lines.append(new_line)

with open(src, "w") as f:
    f.writelines(new_lines)



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


