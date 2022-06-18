import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Som():
    def __init__(self, nx = None, ny = None, preset_map = []):
        self.nx = nx
        self.ny = ny
        self.map = preset_map
    
    def init_map(self):
        self.map = np.random.randint(0, 255, size = (self.nx, self.ny, 3), dtype=np.uint8)
        print("Map initiated")
        
    def display_map(self):
        plt.imshow(self.map, interpolation='nearest')
        plt.show()
        
    def calc_dist(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
    def calc_diff(self, pt1, pt2):
        return np.linalg.norm(pt1 - pt2)
    
    def topo_neigh(self, dist, sigma):
        return np.exp(-(dist**2) / (2*(sigma**2)))
    
    def train(self, input_clrs, num_epoch, alpha_0, sigma_0):
        # normalize
        self.map = self.map / 255
        # start training, iter each epoches
        for epoch in range(num_epoch):
            if (epoch % 10 == 0):
                print("epoch number: " + str(epoch))
            # iter each color
            for clr in input_clrs:
                # find winner coord
                dists = np.array([self.calc_diff(item, clr) for row in self.map for item in row]).reshape(self.nx, self.ny)
                winners = np.where(dists == np.amin(dists))
                win_coord = (np.random.choice(winners[0]), np.random.choice(winners[0])) # randomly pick a winner if there are more than one
                # update weights
                sigma = sigma_0 * np.exp(-epoch / num_epoch)
                alpha = alpha_0 * np.exp(-epoch / num_epoch)
                # 以下代码运行起来肯定贼**慢，懒得改了
                for i in range(self.map.shape[0]):
                    for j in range(self.map.shape[1]):
                        dist = self.calc_dist(win_coord[0], win_coord[1], i, j)
                        topo = self.topo_neigh(dist, sigma)
                        R_dif = self.map[win_coord[0], win_coord[1]][0] - self.map[i, j][0]
                        G_dif = self.map[win_coord[0], win_coord[1]][1] - self.map[i, j][1]
                        B_dif = self.map[win_coord[0], win_coord[1]][2] - self.map[i, j][2]
                        self.map[i, j][0] = self.map[i, j][0] + alpha * topo * R_dif
                        self.map[i, j][1] = self.map[i, j][1] + alpha * topo * G_dif
                        self.map[i, j][2] = self.map[i, j][2] + alpha * topo * B_dif
                        # print("dist, topo, R, G, B diff: " + str(dist) + ' ' + str(topo) + ' ' + str(R_dif) + ' ' + str(G_dif) + ' ' + str(B_dif))
        # transfer RGB value back
        self.map = (self.map * 255).astype(np.uint8)
        print("training finished")
                
                
                
                
                
                
                

    
    
                