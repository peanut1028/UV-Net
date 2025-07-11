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
# raw_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\v5\total.txt"
# train_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\v5\train.txt"
# val_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\v5\val.txt"

# with open(raw_txt_path, "r") as f:
#     lines = f.readlines()

# import random
# random.shuffle(lines)

# train_num = int(len(lines) * 0.9)

# with open(train_txt_path, "w") as f:
#     f.writelines(lines[:train_num])

# with open(val_txt_path, "w") as f:
#     f.writelines(lines[train_num:])


# # 去掉最后一个输入特征
# src = r"E:\Project\AutoPricing\datasets\atwcad\train.txt"

# with open(src, "r") as f:
#     lines = f.readlines()

# new_lines = []
# for l in lines:
#     code, varstr = l.strip().split("  ")
#     values = varstr.split(' ')
#     vars, label = values[:-2], values[-1]
#     new_line = code + "  " + " ".join(vars) + " " + label + "\n"
#     new_lines.append(new_line)

# with open(src, "w") as f:
#     f.writelines(new_lines)


# import os

# raw_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\v4_1\total.txt"
# dst_path = r"E:\Project\AutoPricing\datasets\atwcad\v4_1"
# ref_txt_path = r"E:\Project\AutoPricing\datasets\atwcad\v3_1\val.txt"

# with open(raw_txt_path, "r") as f:
#     lines = f.readlines()

# with open(ref_txt_path, "r") as f:
#     ref_lines = f.readlines()

# ref = []
# for l in ref_lines:
#     code, _ = l.strip().split("  ")
#     ref.append(code)

# train_lines = []
# test_lines = []
# # 需严格匹配，以跟之前的基准对齐
# for c in ref:
#     for l in lines:
#         if not l.strip():
#             continue
#         if c in l:
#             test_lines.append(l)
#             break
    
# for l in lines:
#     if not l.strip():
#         continue
#     if l not in test_lines:
#         train_lines.append(l)

# with open(os.path.join(dst_path, "train.txt"), "w") as f:
#     f.writelines(train_lines)

# with open(os.path.join(dst_path, "test.txt"), "w") as f:
#     f.writelines(test_lines)


"""
bin文件查看，找到face_type和edge_type集合
"""
# import dgl
# import os
# import numpy as np

# bin_dir = r"E:\Project\AutoPricing\datasets\atwcad\bin"
# face_types = []
# edge_types = []
# for file in os.listdir(bin_dir):
#     if file.endswith(".bin"):
#         g = dgl.load_graphs(os.path.join(bin_dir, file))[0][0]
#         t = g.ndata['x'][:, 0, 0].flatten()
#         face_types += np.unique(t).tolist()
#         e = g.edata['x'][:, 0, 0].flatten()
#         edge_types += np.unique(e).tolist()
# print(set(face_types))
# print(set(edge_types))


"""
pt Data查看，找出最大度数
"""
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
import os

# 假设图文件存储在一个目录中，每个文件是一个Data对象
directory = r'E:\Project\AutoPricing\datasets\atwcad\pt'

# 初始化最大度数为0
max_degree = 0

# 遍历目录中的每个文件
for filename in os.listdir(directory):
    if filename.endswith('.pt'):  # 假设文件是以.pt为扩展名的PyTorch保存文件
        filepath = os.path.join(directory, filename)
        
        # 加载图数据
        data = torch.load(filepath)
        
        # 确保data是Data对象
        if isinstance(data, Data):
            # 计算每个节点的度数
            node_degrees = degree(data.edge_index[1], num_nodes=data.num_nodes)
            
            # 找到当前图的最大度数
            current_max_degree = node_degrees.max().item()
            
            # 更新全局最大度数
            if current_max_degree > max_degree:
                max_degree = current_max_degree

print(f"最大度数为: {max_degree}")
