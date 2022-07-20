
import numpy as np
from math import sqrt
from Functions import *
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from skimage.color import rgb2gray
import scipy
from skimage.transform import resize
from skimage.color import rgb2gray

VERBOSE = 1
# ----

import os

#go to directory where images are
#%% preprocessing , resizing, windowing
#Ödem example, preprocessing procedures
img1= plt.imread("bild1.png")
img1 = img1[10:220,30:195,:3]
img2= plt.imread("bild2.png")
img2 = img2[10:220,30:195,:3]
img3= plt.imread("bild3.png")
img3 = img3[10:220,30:195,:3]
#RGB to grayscale
img1 = rgb2gray(img1)
img2 = rgb2gray(img2)
img3 = rgb2gray(img3)

#resizing
img1=resize(img1,(127,100))
img2=resize(img2,(127,100))
img3=resize(img3,(127,100))


#normalization and windoing
img1 = img1/np.max(img1)
img2 = img2/np.max(img2)
img2a = img2-np.percentile(img2,20)
img2a[img2a<0]=0
img2a=img2a/np.max(img2a)
img3 = img3/np.max(img3)

#mask generation
mask=np.copy(img1)
mask[mask>0.25]=0.5

f=[img1,img2a,img3,mask]
f[2][f[2]<0.4]=0
mini= np.percentile(f[2][f[2]>0],60)
#f[2] = (f[2] -mini)
#f[2][f[2]<0.3]=0
maxi = np.max(f[2])
#f[2][f[2]<0] = 0
f[2] = f[2]/maxi
#f[2][f[2]>0.35]=np.max(f[2])
#%%
gt = [gt2,gt3,gt1]
Lambda = 0.15
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
    if return_energy: en = np.zeros(n_it)
    err = np.zeros(n_it)
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
        error = np.sum((np.asarray(x_tilde[:3])-np.asarray(gt))**2)/3
        plt.subplot(141)
        plt.imshow(x[0])
        plt.subplot(142)
        plt.imshow(x[1])
        plt.subplot(143)
        plt.imshow(x[2])
        plt.subplot(144)
        plt.imshow(x[3])
        plt.show()
        if return_energy:
            fid=[]
            tv=[]
            for i in range(0,len(f)):
                fidelity = norm1(Fid1(x_tilde[i], f[i])) + norm1(Fid2(x_tilde[i],f[i]))
                total = Lambda*norm1(gradient(x_tilde[i]))
                fid.append(fidelity)
                tv.append(total)
            energy = 1.0*(np.sum(fid)) + np.sum(tv)
            en[k] = energy
            err[k]=error
            if (VERBOSE and k%10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e \t error %e" %(k,energy,np.sum(fid),np.sum(tv),error))
    if return_energy: return en, x_tilde,err
    else: return x, r,q

p_1 = multi_channel(f,300)

r = np.arange(0,100,1)
plt.plot(r,p1[2])
plt.xlim([0.00, 100])
plt.ylim([100, 750])
plt.show()
