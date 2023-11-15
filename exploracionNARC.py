import numpy as np
import nibabel as nb
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import json as js
from skimage.measure import shannon_entropy as entropy
import pandas as pd

PATH_DATA = Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/NACC/'
PATH_IMAGES = PATH_DATA / 'Images'
PATH_NEW_IMAGES = PATH_DATA / 'NEW_IMAGES'
PATH_TMP = PATH_DATA / 'TMP'
path_csv = PATH_DATA / 'rdd-imaging.csv'
path_csv2 = PATH_DATA / 'uds3-rdd.csv'
os.environ["FREESURFER_HOME"] = "/usr/local/freesurfer"
path_watershed = os.environ["FREESURFER_HOME"] + "/bin/mri_watershed "


def createnewImage(path_data, new_pname, my_img, names):
    command = path_watershed + path_data + ' ' + os.path.join(PATH_TMP, new_pname)

    sizes = [my_img.shape[0], my_img.shape[1], my_img.shape[2]]
    max_entropy = [0, 0, 0]
    selected = [0, 0, 0]

    for i in range(sizes[0]):
        imgH = my_img.slicer[i:i + 1, :, :]
        if entropy(imgH) > max_entropy[0]:
            selected[0] = i
            max_entropy[0] = entropy(imgH)

    for i in range(sizes[1]):
        imgC = my_img.slicer[:, i:i + 1, :]
        if entropy(imgC) > max_entropy[1]:
            selected[1] = i
            max_entropy[1] = entropy(imgC)

    for i in range(sizes[2]):
        imgS = my_img.slicer[:, :, i:i + 1]
        if entropy(imgS) > max_entropy[2]:
            selected[2] = i
            max_entropy[2] = entropy(imgS)

    os.system(command)
    my_img = nb.load(os.path.join(PATH_TMP, new_pname))
    nb.save(my_img.slicer[:, :, selected[2]:selected[2] + 1], os.path.join(PATH_NEW_IMAGES, names[2]))
    nb.save(my_img.slicer[:, selected[1]:selected[1] + 1, :], os.path.join(PATH_NEW_IMAGES, names[1]))
    nb.save(my_img.slicer[selected[0]:selected[0] + 1, :, :], os.path.join(PATH_NEW_IMAGES, names[0]))


def getId(filename):
    idx1 = filename.find('NACC')
    idx2 = filename[idx1:].find('_')
    id = filename[idx1:idx1 + idx2]
    return id


def unzip(file):
    if not os.path.isdir(PATH_TMP):
        os.mkdir(PATH_TMP)
    command = f"unzip {PATH_IMAGES / file} -d {PATH_TMP}"
    os.system(command)


def getDiagnosis(data, df, id):
    diag = df1[
        (df['NACCID'] == id) & (df['NACCADC'] == int(data['InstitutionName'])) & (df['PACKET'] == 'I')][
        'NACCUDSD'].values[0]
    if diag == 1:
        return 'NORMAL'
    elif diag == 2:
        return 'MCI-C'
    elif diag == 3:
        return 'MCI'
    elif diag == 4:
        return 'DEMENTED'
    # La variable 'NACCUDS' devuelve un número indicando el grado de demencia de una persona


if not os.path.isdir(PATH_NEW_IMAGES):
    os.mkdir(PATH_NEW_IMAGES)
os.system(f'rm -rf {PATH_TMP}')
counter = 0
df2 = pd.read_csv(path_csv, dtype='unicode')
df1 = pd.read_csv(path_csv2, dtype='unicode')
# Iteramos sobre todos los archivos comprimidos de la base de datos.
listdir = os.listdir(PATH_IMAGES)
for filezip in listdir:
    naccID = getId(filezip)
    if len(naccID) > 0:
        unzip(filezip)
        for dirname, dirnames, filenames in os.walk(PATH_TMP):
            for filename in filenames:
                if filename.endswith('.nii'):
                    my_img = nb.load(os.path.join(dirname, filename))
                    f = open(os.path.join(dirname, filename[:-3] + 'json'))
                    nii_data = my_img.get_fdata()
                    data = js.load(f)
                    diag = getDiagnosis(data, df1, naccID)
                    new_name = f'{counter}Pa{naccID[4:]}_Di{diag}'
                    new_pname = filename[:-4] + '_stripped.nii'
                    names = [new_name + '_Horizontal.nii', new_name + '_Coronal.nii', new_name + '_Sagital.nii']
                    createnewImage(os.path.join(dirname, filename), new_pname, my_img, names)

                    f.close()
                    counter += 1
        os.system(f'rm -rf {PATH_TMP}')

# Por cada archivo comprimido que  contenga identificador realizamos las siguientes acciones:
#   - descomprimimos
#   - obtenemos MRI
#   - generamos nuevo nombre
#   - skull stripping
#   - guardamos resultado en otra carpeta
#   - borramos descompresión.


# /for file in listdir:
#     if df1[df1['NACCID']==getId(file)].empty or df2[df2['NACCID']==getId(file)].empty or df2[df2['NACCID']==getId(
#     file)]['NACCMRFI'].empty or (df2[df2['NACCID']==getId(file)]['MRIT1'].empty and df2[df2['NACCID']==getId(
#     file)]['MRIT2'].empty) :
#         counter+=1

# for file in listdir:
#     a = file.find('ni.zip')
#     if len(df2[df2['NACCMRFI']==(file[:a]+'.zip')])>0:
#         counter+=1
