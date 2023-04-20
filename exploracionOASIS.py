import numpy as np
import nibabel as nb
import os
from pathlib import  Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


## ARCHIVOS DEL ADNI
PATH_DATA = '/run/media/dazckel/Datos/DatosTFG/OASIS'

for dirname, dirnames, filenames in os.walk(PATH_DATA):
    for filename in filenames:
        my_img  = nb.load(os.path.join(dirname, filename))
        nii_data = my_img.get_fdata()
        nii_aff  = my_img.affine
        nii_hdr  = my_img.header
        print(nii_aff ,'\n',nii_hdr)
        print(nii_data.shape)
        if(len(nii_data.shape)==3):
            for i in range(nii_data.shape[2]):
                plt.imshow(nii_data[:,:, i])
                plt.show()
        if(len(nii_data.shape)==4):
            for frame in range(nii_data.shape[3]):
                for slice_Number in range(nii_data.shape[2]):
                   plt.imshow(nii_data[:,:,slice_Number,frame])
                   plt.show()
