import os
import numpy as np
import torch
from occwl.graph import face_adjacency
from occwl.io import load_step
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRep import BRep_Tool
from tqdm import tqdm
import multiprocessing
from multiprocessing.pool import Pool
import signal
from loguru import logger

from torch_geometric.data import Data  # 替换 DGL 为 PyG 的 Data 类



def list_files(path, exts=['.SLDDRW', '.slddrw', '.SLDPRT', '.sldprt']):
    '''
    list all files fited in exts
    '''
    fileList = os.listdir(path)
    output = []
    for file in fileList:
        name, ext = os.path.splitext(file)
        if ext in exts and name[0] not in ['~', '$']:
            output.append(file)
    return output


class StepConverter(object):
    def __init__(self, 
                numProcesses=1,
                timeLimit=60):
        self.numProcesses = numProcesses
        self.timeLimit = timeLimit

    def build_graph(self, solid):
        # Build face adjacency graph with B-rep entities as node and edge features
        graph = face_adjacency(solid)
        # Compute the UV-grids for faces
        graph_face_feat = []
        for face_idx in graph.nodes:
            # Get the B-rep face
            face = graph.nodes[face_idx]["face"]

            # Get face type(GeomAbs_SurfaceType_Enum: int)
            face_type = face.surface_type_enum()

            # Get face area(float)
            face_area = face.area()

            # Get face loop number(int)
            face_loop_num = face.num_wires()

            topoFace = face.topods_shape()
            # Get face orientation(1 or 0)
            face_orientation = topoFace.Orientation()
            # Get face closed or not(1 or 0)
            face_closed = int(topoFace.Closed())
            # Get face's perimeter(float)
            props = GProp_GProps()
            brepgprop.LinearProperties(topoFace, props)
            face_perimeter = props.Mass()

            # Get face's location
            topoloc = topoFace.Location()
            trsf = topoloc.Transformation()
            # Rotation matrix components
            r11 = trsf.Value(1, 1)
            r12 = trsf.Value(1, 2)
            r13 = trsf.Value(1, 3)
            r21 = trsf.Value(2, 1)
            r22 = trsf.Value(2, 2)
            r23 = trsf.Value(2, 3)
            r31 = trsf.Value(3, 1)
            r32 = trsf.Value(3, 2)
            r33 = trsf.Value(3, 3)
            # Translation components
            dx = trsf.TranslationPart().X()
            dy = trsf.TranslationPart().Y()
            dz = trsf.TranslationPart().Z()

            # concatenate face features
            face_feat = np.array(
                                [[face_type, face_area, face_loop_num],
                                [face_orientation, face_closed, face_perimeter],
                                [r11, r21, r31],
                                [r12, r22, r32],
                                [r13, r23, r33],
                                [dx, dy, dz]]
                                )
            graph_face_feat.append(face_feat)

        graph_edge_feat = []
        for edge_idx in graph.edges:
            # Get the B-rep edge
            edge = graph.edges[edge_idx]["edge"]
            # Ignore degenerate edges, e.g. at apex of cone
            if not edge.has_curve():
                continue
            # Get edge type(int)
            edge_type = edge.curve_type_enum()
            # Get edge length(float)
            edge_length = edge.length()
            # Get edge's orientation(1 or 0)
            topoEdge = edge.topods_shape()
            edge_orientation = topoEdge.Orientation()
            # Get edge's convexity(1 or 0)
            edge_convexity = int(topoEdge.Convex())
            # other parameters 
            _, p1, p2 = BRep_Tool.Curve(topoEdge)
            # Get edge's location
            topoloc = topoEdge.Location()
            trsf = topoloc.Transformation()
            # Rotation matrix components
            r11 = trsf.Value(1, 1)
            r12 = trsf.Value(1, 2)
            r13 = trsf.Value(1, 3)
            r21 = trsf.Value(2, 1)
            r22 = trsf.Value(2, 2)
            r23 = trsf.Value(2, 3)
            r31 = trsf.Value(3, 1)
            r32 = trsf.Value(3, 2)
            r33 = trsf.Value(3, 3)
            # Translation components
            dx = trsf.TranslationPart().X()
            dy = trsf.TranslationPart().Y()
            dz = trsf.TranslationPart().Z()

            # concatenate edge features
            edge_feat = np.array(
                                [[edge_type, edge_length, edge_orientation],
                                [p1, p2, edge_convexity],
                                [r11, r21, r31],
                                [r12, r22, r32],
                                [r13, r23, r33],
                                [dx, dy, dz]]
                                )
            graph_edge_feat.append(edge_feat)

        graph_face_feat = np.asarray(graph_face_feat) # (num_faces, 6, 3)
        graph_edge_feat = np.asarray(graph_edge_feat) # (num_edges, 6, 3)

        # Convert face-adj graph to PyG Data format
        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)  # Shape: [2, num_edges]

        x = torch.from_numpy(graph_face_feat).float()  # Node features
        edge_attr = torch.from_numpy(graph_edge_feat).float()  # Edge features

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data
    
    def process_one_file(self, args):
        file, save_path = args
        code, name = os.path.basename(file).split(" ", 1)
        save_file = os.path.join(save_path, code + ".pt")
        if os.path.exists(save_file):
            return
        try:
            solid = load_step(file)[0]  # Assume there's one solid per file
            data = self.build_graph(solid)
            torch.save(data, save_file)
        except Exception as e:
            logger.exception(f"Processing of file {file} failed")

    def step2graph(self, step_path, save_path):
        os.makedirs(save_path, exist_ok=True)
        step_files = list_files(step_path, [".step", ".STEP"])
        with Pool(processes=self.numProcesses, initializer=initializer) as pool:
            results = []
            try:
                for fn in step_files:
                    result = pool.apply_async(self.process_one_file, 
                                                args=((os.path.join(step_path, fn), save_path),))
                    results.append((result, fn))
                for res, fn in tqdm(results):
                    try:
                        res.get(timeout=self.timeLimit)
                    except multiprocessing.TimeoutError:
                        logger.error(f"Processing of file {fn} timed out, skip")
                    except Exception as e:
                        logger.error(f"Processing of file {fn} failed")
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
        print(f"Successfully converted {len(os.listdir(save_path))} files.")


class TimeoutError(Exception):
    pass


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == '__main__':
    # 示例调用
    step_file = r"E:\Project\AutoPricing\datasets\temp\r0901 ec.stp"
    save_path = r"E:\Project\AutoPricing\datasets\temp"
    converter = StepConverter(numProcesses=1)
    converter.process_one_file((step_file, save_path))