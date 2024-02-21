#Load the numpy, math and scipy libraries
import numpy as np
from math import factorial
from scipy.spatial import Delaunay

# Area of a triangle
def volume(sim, points):
    return abs(np.linalg.det(points[sim[1:]] - points[sim[0]])) / factorial(points.shape[1])

# Delaunay tesselation field estimator
class DTFE:
    def __init__(self, points, velocities, m):
        # The Delaunay tesselation
        self.dim = points.shape[1]
        self.delaunay = Delaunay(points)
        self.velocities = velocities
        
        # The density estimate
        self.rho = np.zeros(len(points))
        for sim in self.delaunay.simplices:
            vol = volume(sim, points)
            for index in sim:
                self.rho[index] += vol
        self.rho = (self.dim + 1) * m / self.rho
        
        # The gradients
        self.Drho = np.zeros([len(self.delaunay.simplices), self.dim])
        self.Dv   = np.zeros([len(self.delaunay.simplices), self.dim, self.dim])
        
        for i in np.arange(len(self.delaunay.simplices)):
            sim = self.delaunay.simplices[i]
            p = points[sim] 
            r = self.rho[sim]
            v = velocities[sim]
            A = p[1:] - p[0]
            self.Drho[i] = np.linalg.inv(A) @ (r[1:] - r[0])
            self.Dv[i] = np.linalg.inv(A) @ (v[1:] - v[0])
    
    def density(self, p):
        simplexIndex = self.delaunay.find_simplex(p)
        pointIndex   = self.delaunay.simplices[simplexIndex][0]
        return  self.rho[pointIndex] + self.Drho[simplexIndex] @ (p - self.delaunay.points[pointIndex])
    
    def v(self, p):
        simplexIndex = self.delaunay.find_simplex(p)
        pointIndex   = self.delaunay.simplices[simplexIndex][0]
        return  self.velocities[pointIndex] + self.Dv[simplexIndex] @ (p - self.delaunay.points[pointIndex])

# Bounding Volume Hierarchy
class BoundingVolumeHierarchy:
    def __init__(self, data, box, X, simplices, depth):
        self.left = None
        self.right = None
        self.data = data
        self.box = box
        self.depth = depth
        
        if self.depth != 0:
            self.branch(depth % len(self.box), X, simplices, self.depth - 1)
  
    def branch(self, dim, X, simplices, depth):
        mins = X[simplices[self.data],...,dim].min(axis=1)
        maxs = X[simplices[self.data],...,dim].max(axis=1)
        
        div = self.box[dim].mean()
        L = mins <= div
        R = maxs >= div

        Lbox = np.copy(self.box)
        Lbox[dim] = np.array([Lbox[dim, 0], div])
        Rbox = np.copy(self.box)
        Rbox[dim] = np.array([div, Rbox[dim, 1]])

        data = np.copy(self.data)
        self.data = None
        self.left  = BoundingVolumeHierarchy(data[L], Lbox, X, simplices, depth)
        self.right = BoundingVolumeHierarchy(data[R], Rbox, X, simplices, depth)

    def findCandidateSimplices(self, p):
        dim = self.depth % len(self.box)
        if self.depth == 0:
            return self.data
        elif p[dim] < self.left.box[dim, 1]:
            return self.left .findCandidateSimplices(p)
        elif p[dim] > self.left.box[dim, 1]:
            return self.right.findCandidateSimplices(p)
        else:
            return np.union1d(self.left .findCandidateSimplices(p), 
                              self.right.findCandidateSimplices(p))
    
    def findBox(self, p):
        dim = self.depth % len(self.box)
        if self.depth == 0:
            print('box:', self.box)
        elif p[dim] < self.left.box[dim, 1]:
            self.left.findBox(p)
        elif p[dim] > self.left.box[dim, 1]:
            self.right.findBox(p)
        else:
            self.left .findBox(p)
            self.right.findBox(p)
            
    def findIntersections(self, p, points, simplices):
        candidates = self.findCandidateSimplices(p)
        node_ids = simplices[candidates]

        ori = points[node_ids[:,0]]
        vs = points[node_ids[:,1:]]
        inv_mat = np.linalg.inv(vs - np.repeat(ori, node_ids.shape[1] - 1, axis=0).reshape(vs.shape)).T    
        newp = np.einsum('imk,km->ki', inv_mat, p - ori)
        val = np.all(newp >= 0, axis=1) & np.all(newp <= 1, axis=1) & (np.sum(newp, axis=1) <= 1)
        return candidates[np.nonzero(val)[0]]
    
# Phase-space density estimator
class PhaseSpace:
    def __init__(self, points_init, points, m, depth, box):
        self.dim = points.shape[1]
        self.points = points
        self.simplices = Delaunay(points_init).simplices

        self.BVH = BoundingVolumeHierarchy(
            np.arange(len(self.simplices)), 
            box.astype(float), 
            self.points, self.simplices, self.dim * depth)
        
        self.rho = np.zeros(len(self.simplices), dtype='float64')
        for index, sim in enumerate(self.simplices):
            self.rho[index] = m / volume(sim, self.points) / factorial(self.dim)
    
    def numberOfStreams(self, p):
        return len(self.BVH.findIntersections(p, self.points, self.simplices))

    def density(self, p):
        return self.rho[self.BVH.findIntersections(p, self.points, self.simplices)].sum()
    
# Phase-space Delaunay tesselation field estimator
class PhaseSpaceDTFE:
    def __init__(self, points_init, points, velocities, m, depth, box):
        self.dim = points.shape[1]
        self.points = points
        self.velocities = velocities
        self.simplices = Delaunay(points_init).simplices

        self.BVH = BoundingVolumeHierarchy(
            np.arange(len(self.simplices)), 
            box.astype(float), 
            self.points, self.simplices, self.dim * depth)
        
        # The density estimate
        self.rho = np.zeros(len(points), dtype='float64')
        for sim in self.simplices:
            vol = volume(sim, self.points)
            for index in sim:
                self.rho[index] += vol
        self.rho = (self.dim + 1) * m / self.rho
        
        # The gradients
        self.Drho = np.zeros([len(self.simplices), self.dim], dtype='float64')
        self.Dv   = np.zeros([len(self.simplices), self.dim, self.dim])
        
        for i in np.arange(len(self.simplices)):
            sim = self.simplices[i]
            p = points[sim] 
            r = self.rho[sim]
            v = velocities[sim]
            A = p[1:] - p[0]
            self.Drho[i] = np.linalg.inv(A) @ (r[1:] - r[0])
            self.Dv[i] = np.linalg.inv(A) @ (v[1:] - v[0])
    
    def numberOfStreams(self, p):
        return len(self.BVH.findIntersections(p, self.points, self.simplices))

    def density(self, p):
        simplexIndices = self.BVH.findIntersections(p, self.points, self.simplices)
        
        dens = 0
        for simplexIndex in simplexIndices:
            pointIndex = self.simplices[simplexIndex][0]
            dens = dens + self.rho[pointIndex] + self.Drho[simplexIndex] @ (p - self.points[pointIndex])
        return dens
    
    def v(self, p):
        simplexIndices = self.BVH.findIntersections(p, self.points, self.simplices)
        
        vs = np.zeros([len(simplexIndices), 2])
        for i, simplexIndex in enumerate(simplexIndices):
            pointIndex   = self.simplices[simplexIndex][0]
            vs[i] = self.velocities[pointIndex] + self.Dv[simplexIndex] @ (p - self.points[pointIndex])
        return vs