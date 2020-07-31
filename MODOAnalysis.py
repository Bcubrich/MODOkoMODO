# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:37:21 2019

@author: bcubrich
"""

#%%
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy
import numpy as np
import pandas as pd
import time
import scipy.stats as stats
import scipy.spatial
import matplotlib.pyplot as plt
from scipy.special import factorial
import itertools



##############################################################################
#                            Dimension Key Vars                              #
##############################################################################
#global neighbors
#global bins
#global s_fact
#global p
#global theta
#
#
#
#neighbors=20   #number of nearest neighbors to look at, need to use user input on this
#bins=[0,30,60,90,120,150,180]
#np.random.seed(123)
#s_fact=factorial(neighbors)
#p=(1/6)**neighbors
#theta=4      #beta angle
##############################################################################


##############################################################################
#                            Basic Functions                                 #
##############################################################################

def do_kdtree(grid_xy,points, n):
    #This function gets the kd tree for the sample, then returns the closest
    #points to each grid node as indices
    points=points[:2,:]
    points=list(points.transpose())
    mytree = scipy.spatial.cKDTree(points)
    dist, indexes = mytree.query(grid_xy, k=n, n_jobs=-1)
    return dist, indexes






###############################################################################
#Random Sample
#np.random.seed(123)
#sample_size=2500
#
#x = list(np.random.uniform(0,1,sample_size))
#y = list(np.random.uniform(0,0.5,sample_size))
#strikes=list(np.random.uniform(15,45,sample_size))
#
#sample_size2=2500
#x.extend(np.random.uniform(0,1,sample_size2))
#y.extend(np.random.uniform(0,0.5,sample_size2))
#strikes.extend(np.random.uniform(0,180,sample_size2))
#
#
###############################################################################
#
#sample=np.vstack((x,y,strikes))
#sample_test=np.std(sample[2,:])
#test=pd.DataFrame(sample, index=['x','y','strike']).T
#test.to_csv('U:/PLAN/BCUBRICH/Python/Fracture MODO/sample.txt', index=False)







##############################################################################
#                            Create Grid                                     #
##############################################################################

#class sample_stat(sample):
#    #use this to get some stats on the sample
#    def __init__(self):
#        self.sample = sample
#        x=self.sample[0]
#        y=self.sample[1]
#        self.min_x=min(x)
#        self.min_y=min(y)
#        self.max_x=max(x)
#        self.max_y=max(y)
#        self.min_gx=min(x)*0.990099
#        self.min_gy=min(y)*0.990099
#        self.max_gx=max(x)*1.01
#        self.max_gy=max(y)*1.01



def analysis(sample, neighbors, bins, theta,grid_n, grid_xy):
    start = time.time()
    bins=[x for x in bins if x<=180]
    p=(1/6)**neighbors
    
    s_fact=factorial(neighbors)
    x=sample[0]
    y=sample[1]
    min_x=min(x)
    min_y=min(y)
    max_x=max(x)
    max_y=max(y)
    min_gx=min_x*0.990099
    min_gy=min_y*0.990099
    max_gx=max_x*1.01
    max_gy=max_y*1.01
    
    
    
    nx=int(np.sqrt(grid_n))
    dx=(max_gx-min_gx)/nx
    dy=dx
    ny=int(np.around((max_gy-min_gy)/dx))
    
    
    
    
    #------------------------------------------------------------------------------
    #This next line is very important, it uses permuations to get the grid locs
#    grid_xy = np.array(list(itertools.product(
#            np.arange(min_gx,max_gx,step=dx),np.arange(min_gy,max_gy,step=dy)))) 
    
    

    dist, indexes=do_kdtree(grid_xy,sample, neighbors)
    
    strikes=sample[2,:]
    
    
    
    mult=[]
    beta=[]
    
    for index in indexes:
        n_strikes=np.take(strikes, index)
        hist1=np.histogram(n_strikes,bins=bins)[0]
        if len(hist1)==7:
            hist1=[hist1[0]+hist1[6],hist1[1],hist1[2],hist1[3],hist1[4],hist1[5]]
        prod_Xi=int(np.prod(factorial(hist1)))
        mult.append(-np.log10(s_fact/prod_Xi*p))
        n_strikes.sort()
        count=0
        for strike1, strike2 in zip(n_strikes,n_strikes[1:]):
            if strike2-strike1<theta: count+=1
        beta.append(count/(neighbors-1))
    
    
    
    output_df=pd.DataFrame([grid_xy[:,0],grid_xy[:,1], mult, beta, np.mean(dist, axis=1),
                            (neighbors/np.max(dist, axis=1)),np.max(dist, axis=1)], index=[
                                    'x','y','Multinomial', 'Beta',
                                    'Avg. Dist. to Neighbor',
                                    'Fracture Density','Max Dist']).T
    
    end = time.time()
#    print ('Completed in: ',end-start)
    print('Number of Grid Nodes:{}, Complection Time:{}'.format(len(grid_xy),end-start))
    
    return output_df

neighbors=20   #number of nearest neighbors to look at, need to use user input on this
bins=[0,30,60,90,120,150,180]
#bins=[0,15,30,60,90,120,150,180]
grid_n=10000
bins=[0,15,45,75,105,135,165,180]
theta=4      #beta angle

#output_df=analysis(sample, neighbors, bins, theta,grid_n)
#output_df.to_csv('U:/PLAN/BCUBRICH/Python/Fracture MODO/output.csv')
#def plot():
#    plt.figure()
#    plt.scatter(output_df.x,output_df.y, s=2)
#    plt.scatter(sample[0], sample[1], s=1, color='r')
#    plt.show()

#plot()

class summary_stats(object):
    #use this to get some stats on the results
    def __init__(self, df):
        self.df = df
        self.spearman_mult_beta=stats.spearmanr(self.df.Multinomial.values, self.df.Beta.values)
        self.pearson_mult_beta=stats.pearsonr(self.df.Multinomial.values, self.df.Beta.values)
        self.spearman_frac_mult=stats.spearmanr(self.df['Fracture Density'].values, self.df.Multinomial.values)
        self.pearson_frac_mult=stats.pearsonr(self.df['Fracture Density'].values, self.df.Multinomial.values)
        self.spearman_frac_beta=stats.spearmanr(self.df['Fracture Density'].values, self.df.Beta.values)
        self.pearson_frac_beta=stats.pearsonr(self.df['Fracture Density'].values, self.df.Beta.values)







#summary_stats(output_df).spearman_mult_beta




###############################################################################
########                                                                #######
########                         GUI                                    #######
########                                                                #######        
###############################################################################   



#In another python file for now




###############################################################################
#                              Unused snippets                              ###
###############################################################################


############################################
#Old way to get grid

#grid_x=[]
#grid_y=[]
#for i in range(nx):
#    grid_x.extend(np.arange(min_x,max_x,step=dx))
#    grid_y.extend([(min_y+dy*i)]*nx)
#    
#combined_x_y_arrays  =np.vstack((grid_x,grid_y)).T
################################################################


######################################################################
#---------------------------------------------------------------------
#Another way to get the summary statistics
#---------------------------------------------------------------------

#def summary_stats(df):
#    #use this to get some stats on the results
#    spearman_mult_beta=stats.spearmanr(df.Multinomial.values, df.Beta.values)
#    pearson_mult_beta=stats.pearsonr(df.Multinomial.values, df.Beta.values)
#    spearman_frac_mult=stats.spearmanr(df['Fracture Density'].values, df.Multinomial.values)
#    pearson_frac_mult=stats.pearsonr(df['Fracture Density'].values, df.Multinomial.values)
#    spearman_frac_beta=stats.spearmanr(df['Fracture Density'].values, df.Beta.values)
#    pearson_frac_beta=stats.pearsonr(df['Fracture Density'].values, df.Beta.values)
#    
#    
#    
#    return {'spearman_mult_beta':spearman_mult_beta[0], 
#            'pearson_mult_beta':pearson_mult_beta[0], 
#            'spearman_frac_mult':spearman_frac_mult[0], 
#            'pearson_frac_mult':pearson_frac_mult[0], 
#            'spearman_frac_beta':spearman_frac_beta[0], 
#            'pearson_frac_beta':pearson_frac_beta[0]}