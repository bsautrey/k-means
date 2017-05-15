# Implement k-means clustering from Andrew Ng's CS229 course: http://cs229.stanford.edu/notes/cs229-notes7a.pdf.

import random,pickle
from copy import copy
from math import fmod

import numpy as np
import matplotlib.pyplot as plot

# tol - stopping criteria. in particular, if the locations of kmeans no longer changes by tol, stop.
# kmeans - the means.
# memberships - what cluster does each x in X belong to?


class KMeans():
    
    def __init__(self):
        self.X = None
        self.tol = None
        self.kmeans = None
        self.memberships = None
        
    def set_X(self,X):
        self.X = X
        
    def set_tolerance(self,tol=0.1):
        self.tol = tol
        
    def initialize_kmeans(self,number_of_clusters):
        self.kmeans = {}
        for inx in xrange(number_of_clusters):
            self.kmeans[inx] = []
            
        # build hypercube from min/max of dataset
        number_of_rows = self.X.shape[0]
        number_of_columns = self.X.shape[1]
        mins = [float('inf'),]*number_of_columns
        maxes = [float('-inf'),]*number_of_columns
        for i in xrange(number_of_rows):
            x = self.X[i,:]
            for j in xrange(number_of_columns):
                val = x[j]
                current_min = mins[j]
                current_max = maxes[j]
                
                if val < current_min:
                    mins[j] = val
                
                if val > current_max:
                    maxes[j] = val 
        
        # select a random mean from the min/max hypercube
        for inx in self.kmeans:
            mean = []
            for j in xrange(number_of_columns):
                min_j = mins[j]
                max_j = maxes[j]
                rand = random.uniform(min_j,max_j)
                mean.append(rand)
                
            mean = np.array(mean)
            mean.resize((1,number_of_columns))
            self.kmeans[inx] = mean[0]
            
    def _initialize_memberships(self):
        self.memberships = {}
        for inx in self.kmeans:
            self.memberships[inx] = []
        
    def run_kmeans(self):
        number_of_rows = self.X.shape[0]
        while True:
            # find minimum distance between each point and a mean. assign point a class membership with minimum distance to mean.
            self._initialize_memberships()
            for i in xrange(number_of_rows):
                x = self.X[i,:]
                min_dist = float('inf')
                min_inx = None
                for inx in self.kmeans:
                    mean = self.kmeans[inx]
                    dist = self._distance(x,mean)
                    if dist < min_dist:
                        min_dist = dist
                        min_inx = inx
                
                self.memberships[min_inx].append(x)
            
            # calculate new kmeans using class memberships. copy old kmeans before updating new kmeans.
            old_kmeans = copy(self.kmeans)
            for inx in self.memberships:
                x_list = self.memberships[inx]
                new_mean = self._calculate_mean(x_list)
                self.kmeans[inx] = new_mean
                
            # calculate convergence by comparing old kmeans to new kmeans, i.e. have the means stopped changing?
            diffs = []
            for inx in self.kmeans:
                old_mean = old_kmeans[inx]
                new_mean = self.kmeans[inx]
                diff = self._distance(old_mean,new_mean)
                diffs.append(diff)
                
            # check if tolerance has been met. if so, calculate meberships/distortion one more time to accounts for last kmeans update.
            max_diff = max(diffs)
            print '\tMAX DIFF:',max_diff
            if max_diff < self.tol:
                distortion = 0.0
                self._initialize_memberships()
                for i in xrange(number_of_rows):
                    x = self.X[i,:]
                    min_dist = float('inf')
                    min_inx = None
                    for inx in self.kmeans:
                        mean = self.kmeans[inx]
                        dist = self._distance(x,mean)
                        if dist < min_dist:
                            min_dist = dist
                            min_inx = inx
                
                    distortion = distortion + min_dist
                    self.memberships[min_inx].append(x)
                
                return distortion
                
    def _calculate_mean(self,x_list):
        if x_list:
            mean = x_list[0]
            for x in x_list[1:]:
                mean = mean + x
        
            l = len(x_list)
            mean = mean/l
            
        # if there are no x's close enough to a mean to have membership in the cluster, reset the mean (bit of a hack).
        else:
            number_of_rows = self.X.shape[0]
            for inx in self.kmeans:
                rand = random.sample(range(number_of_rows),3)
                for i in rand:
                    x = self.X[i,:]
                    mean = x
        
        return mean
        
    # square of distance
    def _distance(self,x_1,x_2):
        diff = x_1 - x_2
        distance = np.dot(diff,diff.T)
        return distance
                
    def generate_example(self,number_of_clusters=5,sample_size_per_cluster=150,number_of_runs=5):
        # assemble data
        X = np.array([])
        X.resize((0,2))
        for inx in xrange(number_of_clusters):
            x_1 = random.sample(range(0,15),1)[0]
            x_2 = random.sample(range(0,15),1)[0]
            cov = random.uniform(-0.75,0.75)
            mean = np.array([x_1,x_2])
            cov = np.array([[1,cov],[cov,1]])
            res = np.random.multivariate_normal(mean,cov,sample_size_per_cluster)
            X = np.row_stack((X,res))
            
        # run kmeans a few times
        distortions = []
        for i in xrange(number_of_runs):
            print 'RUN:',i
            self.set_X(X)
            self.set_tolerance()
            self.initialize_kmeans(number_of_clusters)
            distortion = self.run_kmeans()
            s_memberships = pickle.dumps(self.memberships)
            s_kmeans = pickle.dumps(self.kmeans)
            triple = (distortion,s_memberships,s_kmeans)
            distortions.append(triple)
            print '---\n'
            
        # select run with lowest distortion
        distortions.sort()
        s_memberships = distortions[0][1]
        s_kmeans = distortions[0][2]
        memberships = pickle.loads(s_memberships)
        kmeans = pickle.loads(s_kmeans)
        
        # plot points and color them by class membership
        colors = ['red','orange','green','purple','grey']
        X_clusters = np.array([])
        X_clusters.resize((0,2))
        X_colors = []
        for inx in memberships:
            color = colors[int(fmod(inx,len(colors)))]
            x_list = memberships[inx]
            for x in x_list:
                X_clusters = np.row_stack((X_clusters,x))
                X_colors.append(color)           
                
        X_colors = np.array(X_colors)
        X_colors.resize((number_of_clusters*sample_size_per_cluster,1))
               
        # plot kmeans 
        plot.scatter(X_clusters[:,0],X_clusters[:,1],color=X_colors[:,0],s=0.5)
        for inx in kmeans:
            plot.scatter(kmeans[inx][0],kmeans[inx][1])
            
        plot.show()
                 
        