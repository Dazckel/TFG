import numpy as np
import nibabel as nb
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import pandas as pd
from skimage.measure import shannon_entropy as entropy


def getDifferentImages(datapath):
    ff = []
    for dirname, dirnames, filenames in os.walk(PATH_DATA):
        for filename in filenames:
            ff.append(filename)

    print(f'Imágenes en total: {len(ff)}')
    print(f'Imágenes sin repetir: {len(set(ff))}')


### A continuación se definen funciones para extraer información.
def getInformation(filename):
    d = filename.find('S')
    patienID = filename[d - 4:d + 6]
    d = filename.find('_I')
    p = filename.find('.')
    imgID = filename[d + 1:p]
    d = filename.find('Br_') + 3
    fecha = filename[d:d + filename[d:].find('_')]
    return [patienID, imgID, fecha]


## ARCHIVOS DEL ADNI
PATH_DATA_TMP = Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/ADNI/TMP'
PATH_DATA = Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/ADNI/Images'
PATH_INFORMATION = Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/ADNI/Informacion/DatosImagenes'
csv_files = os.listdir(PATH_INFORMATION)
# Atributos para una imagen:
# - ID Paciente
# - ID Imagen.
# - Fecha
getDifferentImages(PATH_DATA)
# La siguiente linea es clave para poder ejecutar el skull stripping desde aqui
os.environ["FREESURFER_HOME"] = "/usr/local/freesurfer"
path_watershed = os.environ["FREESURFER_HOME"] + "/bin/mri_watershed "

for dirname, dirnames, filenames in os.walk(PATH_DATA):
    for filename in filenames:
        if not ('___Di' in filename):
            _, idAUX, _ = getInformation(filename)
            path_data = os.path.join(dirname, filename)
            my_img = nb.load(path_data)
            # Nos quedamos con las imágenes del centro.
            patienID, imgID, fecha = getInformation(filename)
            for fi in csv_files:
                if (fi[0] != '.'):
                    df = pd.read_csv(os.path.join(PATH_INFORMATION, fi))
                    res = [i for i in df.columns if 'Diagnosis' in i]
                    if (df[df['Image.ID'] == int(imgID[1:])].__len__() > 0):
                        diag = df[df['Image.ID'] == int(imgID[1:])][res].values[0][0]
            new_pname = filename[:-4] + f'___Di{diag}.nii'
            command = path_watershed + path_data + ' ' + os.path.join(dirname, new_pname)
            os.system(command)
            os.system(f"rm {path_data}")

        # if(len(nii_data.shape)==3):
        #     fig, axs = plt.subplots(1,3)
        #     fig.suptitle(f' Imagen ID: {imgID};  Patien: {patienID};\n Fecha: {fecha}; Diagnosis: {diag}')
        #     middle = [nii_data.shape[0] // 2, nii_data.shape[1] // 2, nii_data.shape[2] // 2]
        #     axs[0].imshow(nii_data[middle[0], :, :])
        #     axs[0].set_title('Horizontal')
        #     axs[0].axis('off')
        #     axs[1].imshow(nii_data[:, middle[1], :])
        #     axs[1].set_title('Coronal')
        #     axs[1].axis('off')
        #     axs[2].imshow(nii_data[:,:, middle[2]])
        #     axs[2].set_title('Sagital')
        #     axs[2].axis('off')
        #     plt.show()
