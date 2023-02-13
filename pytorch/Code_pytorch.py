#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:44:23 2023

@author: nadja
"""

from __future__ import division

import os
os.chdir('/media/nadja/BC1048F81048BB62/CV_pytorch')

import numpy as np
from math import sqrt
from Functions_pytorch import *
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from skimage.color import rgb2gray
import scipy
from skimage.transform import resize
from skimage.color import rgb2gray

# ----
VERBOSE = 1
# ----

import os
os.chdir('/media/nadja/BC1048F81048BB62/CV_pytorch')


gt = plt.imread("textur1_GT.png")
gt = gt[:,47:957]
gt =resize(gt,(100,100))
gt1 = gt[:,:,0]
gt1[gt1>0.5]=1
gt1[gt1<1]=0
#gt1 = rgb2gray(gt1[:,:])



gt3 = gt[:,:,2]
gt3[gt3>0.7]=1
gt3[gt3<1]=0
#gt3 = rgb2gray(gt3[:,:])
gt2 = np.ones_like(gt1)- (gt1+gt3)
#gt2 = rgb2gray(gt2[:,:])

GT = [gt3,gt1,gt2]
f=[]
for i in range(1,4):
    img = plt.imread("gabor" + str(i)+".png")
    img =resize(img,(100,100))
    img= rgb2gray(img[:,:,:3])
    img=img/np.max(img)
 #   img = np.expand_dims(img,axis=0)
    f.append(img)
device = 'cpu'
f = torch.tensor(f, dtype = torch.float64).to(device)
Lambda = 0.1
n_it = 2
'''now f is a list'''
def multi_channel(f, n_it, return_energy=True):
    sigma = 1.0/np.sqrt(10)
    tau = 1.0/np.sqrt(10)
    x = f
    p=[]
    #compute gradient over all channels of input lifted torch tensor
    p = gradient(x)    
    q = f
    r = f
    x_tilde=f

    theta = 1.0
   # if return_energy: en = torch.zeros(n_it)
    en= torch.zeros(n_it)
    tvsum = torch.zeros(n_it)
    fidsum = torch.zeros(n_it)
    error = torch.zeros(n_it)
    n_it = 200
    #start iterations
    for k in range(0, n_it):
        ''' Update dual variables of image f'''
        dual1=[]
        dual2=[]
        dual3=[]
        p1 = proj_l1_grad(p + sigma*gradient(x_tilde), Lambda) #update of TV
        q1 = proj_l1(q + sigma*Fid1(x_tilde, f), 1) #Fidelity term without norm (change this in funciton.py)
        r1 = proj_l1(r + sigma*Fid2(x_tilde, f),1)
        p = p1.clone()
        q = q1
        r = r1
        # Update primal variables
        x_old = x_tilde #hier habe ich x durch x_tilde ersetzt
       # x_tilde=[]
        x = proj_unitintervall(x_old + tau*div(p1) - tau*adjoint_der_Fid1(x_old,f,q)- tau*adjoint_der_Fid2(x_old,f,r)) #proximity operator of indicator function on [0,1]
        #x_tilde = x
       # x_tilde = proj_unitball(x_tilde)
        x_tilde = euclidean_proj_simplex(x)
        gay =x_tilde+ theta*(x_tilde-x_old)
        x_tilde = gay
       # err = torch.sum(torch.abs((x_tilde)-(GT)))/x_tilde.shape[0]
        plt.subplot(131)
        plt.imshow(x[0].cpu())
        plt.subplot(132)
        plt.imshow(x_tilde[1].cpu())
        plt.subplot(133)
        plt.imshow(x_tilde[2].cpu())
        #plt.subplot(144)
        #plt.imshow(x[3])
        plt.show()
        # if return_energy:
        #     fid=[]
        #     tv=[]
        #     tv_plot=[]
        #     for i in range(0,len(f)):
        #         fidelity = norm1(Fid1(x_tilde[i], f[i])) + norm1(Fid2(x_tilde[i],f[i]))
        #         total = Lambda*norm1(gradient(x_tilde[i]))
        #         fid.append(fidelity)
        #         tv_p = norm1(gradient(x_tilde[i]))
        #         tv.append(total)
        #         tv_plot.append(tv_p)
        #     sumfid = np.sum(fid)
        #     energy = 1.0*(np.sum(fid)) + np.sum(tv)
        #     tvsum[k] = np.sum(tv_plot)
        #     fidsum[k]=sumfid
        #     en[k] = energy
        #     error[k]=err
      #      if (VERBOSE and k%10 == 0):
       #         print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,np.sum(fid),np.sum(tv)))
   # if return_energy: return en, x_tilde, tvsum, fidsum, error
    #else: return x, r,q
    return x,r,q

p = multi_channel(f,200)
#comparison to normal nonconvex chan vese
Lambda=0.5
q1= multi_channel(f,200)
Lambda=0.1
q2= multi_channel(f,200)

#show the results
r = np.arange(0,200,1)
plt.plot(r,p[0],'g',label="energy")
plt.plot(r,q1[0],'g--')
plt.plot(r,q2[0],'g:')
plt.plot(r,p[2], 'r',label = "TV")
plt.plot(r,q1[2],'r--')
plt.plot(r,q2[2],'r:')
#plt.plot(r,p[3],'c-',label="fidelity")
#plt.plot(r,q1[3],'c--')
#plt.plot(r,q2[3],'c:')
plt.plot(r,p[4], 'b-', label = "error")
plt.plot(r,q1[4], 'b--')
plt.plot(r,q2[4], 'b:')

#plt.plot(r,q005[3],'c--')
plt.xlim([0.00, 200])
plt.ylim([0, 2500])
plt.legend(loc="upper right")
plt.show()






q2[1][0][q2[1][0]<0.33]=0
q2[1][1][q2[1][1]<0.33]=0
q2[1][2][q2[1][2]<0.33]=0
M = np.expand_dims(q2[1][0], axis=-1)
N= np.expand_dims(q2[1][1], axis=-1)
O=np.expand_dims(q2[1][2], axis=-1)
M= np.concatenate((M,N,O), axis = -1)
g= np.argmax(M,axis=-1)
plt.imshow(g)