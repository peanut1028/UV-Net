from occwl.io import load_step
from occwl.graph import face_adjacency

step_file = r"A018HA-01-01-014A 轴承侧挡板.STEP"
solid = load_step(step_file)[0]
print(solid)
graph = face_adjacency(solid)
print(graph)