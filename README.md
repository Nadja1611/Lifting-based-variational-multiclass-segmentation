# Lifting-based-variational-multiclass-segmentation


This repository contains the source code of the paper "Lifting-based variational multiclass segmentation: design, analysis and implementation". 

## Highlights
The main components of the proposed pipeline are as follows:

1. Lifting: Choose $K$ feature enhancing transforms $\Phi_1,\dots,\Phi_K$ in a way thath the intensity values of the $k$-th feature map $\phi_K\coloneqq\Phi_k(f)$ allow to well separate $\Sigma_k$ rom the remaining part $\Omega\setminus\Sigma_k$.

2. Minimization: For given parameter $\lambda>0$, compute a minimizer of the proposed energy functional (see p.2 Problem 1.1 in the paper)
3. Assignment: For each $k\in\{0,\dots,K\}$ define the region $\Sigma_k$ as the set of all $x\in\Omega$ such that $u_k^{\lambda}(x)$ is maximal along the values $u_0^{\lambda}(x),\dots,u_K^{\lambda}(x)$ with $u_0 = 1-\sum_{k=1}^K u_k$.

## Requirements
To install the requirements needed to run the code via anaconda execute:
* conda create --name <env> --file requirements.txt
The required packages are:
 1. scipy=1.7.3
 2. scikit-image=0.19.2
 3. scikit-learn=1.0.2
 4. matplotlib=3.5.1
Note: Other versions of these packages might work too, there are just those we used.
## Code
The code consists of the following parts:
1. functions.py: This script contains all of the functions needed (for a detailed description of these functiosn we refer to Section 3 of the manuscript)
