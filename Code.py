
from __future__ import division
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

# ----
VERBOSE = 1
# ----

import os
os.chdir('D://Chan Vese Algorithm//Code_sauber//Code_Brodatz3')


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
    f.append(img)


Lambda = 0.2
'''now f is a list'''
def multi_channel(f, n_it, return_energy=True):
    f=np.asarray(f)
    sigma = 1.0/np.sqrt(10)
    tau = 1.0/np.sqrt(10)
    x = f
    p=[]
    for i in range(0,len(f)):
        p2 = 1*gradient(x[i]) 
        p.append(p2)
    q = 1*f
    r = 1*f
    x_tilde=f
   # x_tilde = [0.3333333*np.ones_like(f[0]),0.4*np.ones_like(f[0]),0.1*np.ones_like(f[0]),0.7*np.ones_like(f[0]),0.5*np.ones_like(f[0])]
   # x_tilde = [a,b,c,d]
    theta = 1.0
    if return_energy: en = np.zeros(n_it)
    en= np.zeros(n_it)
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
        err = np.sum(np.abs(np.asarray(x_tilde)-np.asarray(GT)))/3
        plt.subplot(131)
        plt.imshow(x[0])
        plt.subplot(132)
        plt.imshow(x[1])
        plt.subplot(133)
        plt.imshow(x[2])
        #plt.subplot(144)
        #plt.imshow(x[3])
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