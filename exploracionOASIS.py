import numpy as np
import nibabel as nb
import os
from pathlib import  Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from pathlib import Path
import pandas as pd

## ARCHIVOS DEL ADNI
PATH_DATA =  Path(os.path.dirname(__file__)).parent /'Datos' /'DatosTFG'/'OASIS' / 'Images'
path_csv = Path(os.path.dirname(__file__)).parent /'Datos' /'DatosTFG'/'OASIS' / 'OASIS3_data_files' \
           / 'UDSd1' / 'csv' / 'OASIS3_UDSd1_diagnoses.csv'
path_csv2 = Path(os.path.dirname(__file__)).parent /'Datos' /'DatosTFG'/'OASIS' / 'OASIS3_data_files' \
           / 'UDSd2' / 'csv' / 'OASIS3_UDSd2_med_conditions.csv'
df1 = pd.read_csv(path_csv)
df2 = pd.read_csv(path_csv2)


for dirname, dirnames, filenames in os.walk(PATH_DATA):
    for filename in filenames:
        my_img  = nb.load(os.path.join(dirname, filename))
        nii_data = my_img.get_fdata()
        nii_aff  = my_img.affine
        nii_hdr  = my_img.header
        if(len(nii_data.shape)==3):
            for i in range(nii_data.shape[2]):
                middle = [nii_data.shape[0] // 2, nii_data.shape[1] // 2, nii_data.shape[2] // 2]
                imgH = nb.Nifti1Image(nii_data[middle[0], :, :], np.eye(4))
                imgC = nb.Nifti1Image(nii_data[:, middle[1], :], np.eye(4))
                imgS = nb.Nifti1Image(nii_data[:, :, middle[2]], np.eye(4))


#Bucle para descomprimir todos las imágenes de la base de datos OASIS.
# for dirname, dirnames, filenames in os.walk(PATH_DATA):
#     for filename in filenames:
#         if filename.endswith('.gz'):
#             os.system('gzip -d ' + os.path.join(dirname,filename))

#Elimamos las imágenes que solo contienen hipocampos

# for dirname, dirnames, filenames in os.walk(PATH_DATA):
#     for filename in filenames:
#         if filename.find('hippocampus')>0:
#             path_to_del = Path(os.path.join(dirname,filename))
#             os.system('rm -r '+ path_to_del.parent.absolute().parent.__str__())


        # if(len(nii_data.shape)==4):
        #     for frame in range(nii_data.shape[3]):
        #         for slice_Number in range(nii_data.shape[2]):
        #            plt.imshow(nii_data[:,:,slice_Number,frame])
        #            plt.show()