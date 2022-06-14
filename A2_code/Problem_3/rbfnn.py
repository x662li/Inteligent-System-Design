from turtle import forward
import numpy as np
import pandas as pd

class Rbfnn():
    def __init__(self, rbf_param = [], W = []):
        self.rbf_param = rbf_param # [[center1, width1], [center2, width2], ...p]
        self.W = W # [w1, w2, ....]
    
    def get_parameters(self):
        pass
    
    def rbf(self, X, center, width):
        norm = np.sqrt((X[0] - center[0])**2 + (X[1] - center[1])**2)
        return np.exp(-norm / (2*width**2))
    
    def compute_G(self, X):
        G = np.zeros((len(X), len(self.rbf_param)))
        for i in range(len(self.rbf_param)):
            for j in range(len(X)):
                G[j, i] = self.rbf(X[j], self.rbf_param[i][0], self.rbf_param[i][1])
        return G
    
    def predict(self, X): # X = [[x1,y1], [x2,y2], ...]
        G = self.compute_G(X)
        return np.dot(G, self.W)
            
    def compute_W(self, G, Y):
        self.W = np.dot(np.linalg.pinv(G), Y)
        
        