# Lifting-based-variational-multiclass-segmentation


This repository contains the source code of the paper "Lifting-based variational multiclass segmentation: design, analysis and implementation". 

## Highlights
The main components of the proposed pipeline are as follows:

1. Lifting: Choose K feature enhancing transforms in a way thath the intensity values of the k-th feature map allow to well separate $R_1$ from the remaining part $\Omega\setminus R_1$.
2. Training of a segmentation module, a U-Net, on the remaining slices to obtain axial probability masks of the PLIC.
3. Training of a classification network, the Thalamic-Slice-Selector, that identifies the slice that corresponds to the level of the boundary between upper and middle third of the thalamus.
4. The combination of the results from 2. and 3. yield ROI-boxes, from which the patches in coronal and sagittal plane view, arise.  
5. Training of a segmentation module on coronal and sagittal plane view patches. 
6. Combination of the resulting probability masks and creation of binary PLIC-masks.
