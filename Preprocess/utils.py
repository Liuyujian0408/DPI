
import numpy as np


def angle(vector1, vector2):
    cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1)*np.linalg.norm(vector2)+1e-8)
    angle = np.arccos(cos_angle)
    # angle2=angle*360/2/np.pi
    return angle

def area_triangle(vector1, vector2):
    trianglearea = 0.5 * np.linalg.norm( \
        np.cross(vector1, vector2))
    return trianglearea

def cal_dist(vertex1, vertex2, ord=2):
    return np.linalg.norm(vertex1 - vertex2, ord=ord)






