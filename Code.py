# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:14:44 2021

@author: c7021086
"""


from __future__ import division
import numpy as np
from math import sqrt
from functions import *
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from skimage.color import rgb2gray
import scipy
from skimage.transform import resize
from skimage.color import rgb2gray

#from utils import *

#from tomo_operators import AstraToolbox
#from convo_operators import *

# ----
VERBOSE = 1
# ----

import os
os.chdir('D://Chan Vese Algorithm//Code_sauber//Code_butterfly')

#Schmetterling example
img = plt.imread("filtering_butterfly.png")[:,:,:3]
img2 = plt.imread("filtering_butterfly2.png")[:,:,:3]
gt1 = plt.imread("gt_butterfly.png")
gt1[gt1[:,:,0]>0.6]=1
gt1[gt1<1]=0
gt2 = plt.imread("gt_flower.png")
gt2[gt2>0.05]=1
gt2[gt2<0.9]=0
img = rgb2gray(img[:,:,:])
img2 = rgb2gray(img2[:,:,:])
img=resize(img,(100,100))
img2=resize(img2,(100,100))
img = img/np.max(img)
img2 = img2/np.max(img2)

gt1 = rgb2gray(gt1[:,:,:3])
gt2 = rgb2gray(gt2[:,:,:3])
gt1=resize(gt1,(100,100))
gt2=resize(gt2,(100,100))
gt1 = gt1/np.max(gt1)
gt2 = gt2/np.max(gt2)

gt =[gt1,gt2]
f=[img, img2]
Lambda = 0.5
'''now f is a list'''
def multi_channel(f, n_it, return_energy=True):
    f=np.asarray(f)
    sigma = 1.0/16.0
    tau = 1.0/8.0
    x = f
    p=[]
    for i in range(0,len(f)):
        p2 = 1*gradient(x[i]) 
        p.append(p2)
    q = 1*f
    r = 1*f
    x_tilde = f
    theta = 1.0
    if return_energy: en= np.zeros(n_it)
    tvsum = np.zeros(n_it)
    fidsum = np.zeros(n_it)
    error = np.zeros(n_it)
    for k in range(0, n_it):
        ''' Update dual variables of image f'''
        dual1=[]
        dual2=[]
        dual3=[]
        for i in range(0,len(f)):
            p1 = proj_l1_grad(p[i] + sigma*gradient(x_tilde[i]), Lambda) #update of TV
            q1 = proj_l1(q[i] + sigma*Fid1(x_tilde[i], f[i]), 1) #Fidelity term without norm (change this in funciton.py)
            r1 = proj_l1(r[i] + sigma*Fid2(x_tilde[i], f[i]),1)
            dual1.append(p1)
            dual2.append(q1)
            dual3.append(r1)
        p = dual1
        q = dual2
        r=dual3
        # Update primal variables
        x_old = x_tilde#hier habe ich x durch x_tilde ersetzt
        x_tilde=[]
        for i in range(0,len(f)):
            #y =x_old[i] + tau*div(p[i]) - tau*adjoint_der_Fid1(x_old[i],f[i],q[i])- tau*adjoint_der_Fid2(x_old[i],f[i],r[i]) 
            #print(np.max(y))
            #x =x_old[i] + tau*div(p[i]) - tau*adjoint_der_Fid1(x_old[i],f[i],q[i])- tau*adjoint_der_Fid2(x_old[i],f[i],r[i]) 

            x = proj_unitintervall(x_old[i] + tau*div(p[i]) - tau*adjoint_der_Fid1(x_old[i],f[i],q[i])- tau*adjoint_der_Fid2(x_old[i],f[i],r[i])) #proximity operator of indicator function on [0,1]
            #x_tilde = proj_unitball(x_tilde)
            x_tilde.append(x)
       # x_tilde = proj_unitball(x_tilde)
        x_tilde = np.asarray(x_tilde)
        x_tilde = np.apply_along_axis(euclidean_proj_simplex, 0, x_tilde)
        x=[]
        for i in range(0,len(f)):
            x_t = x_tilde[i] + theta*(x_tilde[i] - x_old[i])
            x.append(x_t)
        x_tilde=x
        err = np.sum(np.abs(np.asarray(x_tilde)-np.asarray(gt)))
        plt.subplot(121)
        plt.imshow(x[0])
        plt.subplot(122)
        plt.imshow(x[1])
        plt.show()
        if return_energy:
            fid=[]
            tv=[]
            tv_plot=[]
            for i in range(0,len(f)):
                fidelity = norm1(Fid1(x_tilde[i], f[i])) + norm1(Fid2(x_tilde[i],f[i]))
                total = Lambda*norm1(gradient(x_tilde[i]))
                fid.append(fidelity)
                tv_p = norm1(gradient(x_tilde[i]))
                tv.append(total)
                tv_plot.append(tv_p)
            sumfid = np.sum(fid)
            energy = 1.0*(np.sum(fid)) + np.sum(tv)
            tvsum[k] = np.sum(tv_plot)
            fidsum[k]=sumfid
            en[k] = energy
            error[k]=err
            if (VERBOSE and k%10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,np.sum(fid),np.sum(tv)))
    if return_energy: return en, x_tilde, tvsum, fidsum, error
    else: return x, r,q
p = multi_channel(f,200)

Lambda=0.1
q1= multi_channel(f,200)
#show the results
r = np.arange(0,200,1)
plt.plot(r,p[0],'g',label="energy")
plt.plot(r,q1[0],'g--')
plt.plot(r,p[2], 'r',label = "TV")
plt.plot(r,q1[2],'r--')
plt.plot(r,p[3],'c-',label="fidelity")
plt.plot(r,q1[3],'c--')
plt.plot(r,p[4], 'b-', label = "error")
plt.plot(r,q1[4], 'b--')

#plt.plot(r,q005[3],'c--')
plt.xlim([0.00, 200])
plt.ylim([0, 2750])
plt.legend(loc="upper right")
plt.show()


