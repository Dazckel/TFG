import numpy as np
import nibabel as nb
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from pathlib import Path
import pandas as pd


def getId(filename):
    idx1 = filename.find('OAS')
    idx2 = filename[idx1:].find('_')
    id = filename[idx1:idx1 + idx2]
    return id


def getSessionId(filename):
    idx1 = filename.find('d')
    idx2 = filename[idx1:].find('_')
    if idx2 == -1:
        id = filename[idx1:]
    else:
        id = filename[idx1:idx1 + idx2]
    return id


def getDiagnosis(df, oasisID, sessID):
    diag = df[(df['OASISID'] == oasisID) & (df['OASIS_session_label'] == f'{oasisID}_UDSd1_{sessID}')]

    if not diag.empty:
        if diag['NORMCOG'].values == 1:
            return 'NORMCOG'
        elif diag['DEMENTED'].values == 1:
            'DEMENTED'
        elif diag['MCIAMEM'] == 1:
            return 'MCIAMEM'


## ARCHIVOS DEL ADNI
PATH_DATA = Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/OASIS/'
PATH_IMAGES = PATH_DATA / 'NewImages'
path_csv = PATH_DATA / 'OASIS3_data_files' \
           / 'UDSd1' / 'csv' / 'OASIS3_UDSd1_diagnoses.csv'
path_csv2 = PATH_DATA / 'OASIS3_data_files' \
            / 'UDSd2' / 'csv' / 'OASIS3_UDSd2_med_conditions.csv'
df1 = pd.read_csv(path_csv)
df2 = pd.read_csv(path_csv2)
allFiles = os.listdir(PATH_IMAGES)
index = zip(df1['OASISID'].to_list(), df1['OASIS_session_label'].to_list())

counter = 0
for id, ses in index:
    for file in allFiles:
        if (id in file) and (getSessionId(file) in ses):
            counter += 1
            print(counter)

# for dirname, dirnames, filenames in os.walk(PATH_DATA):
#     for filename in filenames:
#         oasisID = getId(filename)
#         sessID = getSessionId(filename)
#         diag = getDiagnosis(df1, oasisID, sessID)
#         print(diag)

# my_img = nb.load(os.path.join(dirname, filename))
# nii_data = my_img.get_fdata()
# nii_aff = my_img.affine
# nii_hdr = my_img.header

# Bucle para descomprimir todos las imágenes de la base de datos OASIS.
# for dirname, dirnames, filenames in os.walk(PATH_DATA):
#     for filename in filenames:
#         if filename.endswith('.gz'):
#             os.system('gzip -d ' + os.path.join(dirname,filename))

# Elimamos las imágenes que solo contienen hipocampos

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
