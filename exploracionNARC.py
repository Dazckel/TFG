import numpy as np
import nibabel as nb
import os
from pathlib import  Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import json as js
import pandas as pd


PATH_DATA = '/run/media/dazckel/Datos/DatosTFG/Disco'
path_csv = '/run/media/dazckel/Datos/DatosTFG/Disco/DatosTFG/investigator_mri_nacc60.csv'
path_csv2 = '/run/media/dazckel/Datos/DatosTFG/Disco/DatosTFG/investigator_nacc60.csv'

df1 = pd.read_csv(path_csv)
# df2 = pd.read_csv(path_csv2)


for dirname, dirnames, filenames in os.walk(PATH_DATA):
    for filename in filenames:
        if filename.endswith('.nii'):
            my_img  = nb.load(os.path.join(dirname, filename))
            nii_data = my_img.get_fdata()
            nii_aff  = my_img.affine
            nii_hdr  = my_img.header
            print(nii_aff ,'\n',nii_hdr)
            print(nii_data.shape)
            if(len(nii_data.shape)==3):
               sh = nii_data.shape
               plt.imshow(nii_data[sh[0]//2,:,: ])
               plt.show()
               plt.imshow(nii_data[:,sh[1]//2,: ])
               plt.show()
               plt.imshow(nii_data[:,:,sh[2]//2 ])
               plt.show()
            if(len(nii_data.shape)==4):
               for frame in range(nii_data.shape[3]):
                   for slice_Number in range(nii_data.shape[2]):
                       plt.imshow(nii_data[:,:,slice_Number,frame])
                       plt.show()
        elif filename.endswith('.json'):
            f = open(os.path.join(dirname, filename))
            data = js.load(f)
            for key in data.keys():
                print(f'{key}: {data[key]}')
            f.close()

