'''
MIT License
Copyright (c) 2019 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.  
'''

import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from collections import deque
from .utils import distance
import math

class Vertex:
    def __init__(self, loc, parent):
        self.loc = loc
        self.parent = None

class Graph:
    ''' Define graph '''
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos:0}
        self.neighbors = {0:[]}
        self.distances = {0:0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))


    def randomPositionBiased(self):
        rx = random()
        ry = random()

        greater_x = max(self.startpos[0], self.endpos[0])
        lesser_x = min(self.startpos[0], self.endpos[0])
        self.sx = greater_x - lesser_x

        greater_y = max(self.startpos[1], self.endpos[1])
        lesser_y = min(self.startpos[1], self.endpos[1])
        self.sy = greater_y - lesser_y

        posx = int(rx*self.sx/2 + lesser_x)
        posy = int(ry*self.sy/2 + lesser_y)

        # posx = int(self.startpos[0] - (self.sx / 2.) + rx * self.sx * 2)
        # posy = int(self.startpos[1] - (self.sy / 2.) + ry * self.sy * 2)
        return posx, posy

    def randomPosition(self):
        rx = random()
        ry = random()
        return int(rx * 2428), int(ry * 2428) 

# class Line():
#     ''' Define line '''
#     def __init__(self, p0, p1):
#         self.p = np.array(p0)
#         self.dirn = np.array(p1) - np.array(p0)
#         self.dist = np.linalg.norm(self.dirn)
#         self.dirn /= self.dist # normalize

#     def path(self, t):
#         return self.p + t * self.dirn


# def Intersection(line, center, radius):
#     ''' Check line-sphere (circle) intersection '''
#     a = np.dot(line.dirn, line.dirn)
#     b = 2 * np.dot(line.dirn, line.p - center)
#     c = np.dot(line.p - center, line.p - center) - radius * radius

#     discriminant = b * b - 4 * a * c
#     if discriminant < 0:
#         return False

#     t1 = (-b + np.sqrt(discriminant)) / (2 * a);
#     t2 = (-b - np.sqrt(discriminant)) / (2 * a);

#     if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
#         return False

#     return True



# def isInObstacle(vex, obstacles, radius):
#     for obs in obstacles:
#         if distance(obs, vex) < radius:
#             return True
#     return False


# def isThruObstacle(line, obstacles, radius):
#     for obs in obstacles:
#         if Intersection(line, obs, radius):
#             return True
#     return False


# def newVertex(randvex, nearvex, stepSize):
#     dirn = np.array(randvex) - np.array(nearvex)
#     length = np.linalg.norm(dirn)
#     dirn = (dirn / length) * min (stepSize, length)

#     newvex = (nearvex[0]+dirn[0], nearvex[1]+dirn[1])
#     return newvex


def window(startpos, endpos):
    ''' Define seach window - 2 times of start to end rectangle'''
    width = endpos[0] - startpos[0]
    height = endpos[1] - startpos[1]
    winx = startpos[0] - (width / 2.)
    winy = startpos[1] - (height / 2.)
    return winx, winy, width, height


def isInWindow(pos, winx, winy, width, height):
    ''' Restrict new vertex insides search window'''
    if winx < pos[0] < winx+width and \
        winy < pos[1] < winy+height:
        return True
    else:
        return False

def dijkstra(G):
    '''
    Dijkstra algorithm for finding shortest path from start position to end.
    '''
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)