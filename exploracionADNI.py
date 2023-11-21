import nibabel as nb
import os
from pathlib import Path
import platform
from deepbrain import Extractor
from skimage.measure import shannon_entropy as entropy
import pandas as pd
import re


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
    imgID = re.sub("[^0-9]", "", imgID)
    d = filename.find('Br_') + 3
    fecha = filename[d:d + filename[d:].find('_')]
    return [patienID, imgID, fecha]

operating_system = platform.system()

if operating_system == 'Windows':
    PATH_DATA = Path('F:/') / 'Dataset/ADNI/NewImages'
    PATH_DATA_TMP = Path('F:/') /'/Dataset/ADNI/TMP'
    PATH_INFORMATION = Path('F:/') / '/Dataset/ADNI/Informacion/DatosImagenes'
    csv_files = os.listdir(PATH_INFORMATION)
elif operating_system == 'Linux':
    PATH_DATA_TMP = Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/ADNI/TMP'
    PATH_DATA = Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/ADNI/Images'
    PATH_INFORMATION = Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/ADNI/Informacion/DatosImagenes'
    csv_files = os.listdir(PATH_INFORMATION)

# Atributos para una imagen:
# - ID Paciente
# - ID Imagen.
# - Fecha
def processData():
    getDifferentImages(PATH_DATA)

    ext = Extractor()

    for dirname, dirnames, filenames in os.walk(PATH_DATA):
        for filename in filenames:
            if len(os.listdir(dirname)) == 1:
                if '.nii' in filename:
                    path_data = os.path.join(dirname, filename)
                    new_path_data = (Path(dirname) / Path(filename[:filename.find('.nii')] )).__str__()
                    img = nb.load(path_data)
                    try:
                        data = img.get_fdata()

                        prob = ext.run(data)

                        mask = prob > 0.5
                        data_stripped = data * mask

                        sizes = [data_stripped.shape[0], data_stripped.shape[1], data_stripped.shape[2]]
                        max_entropy = [0, 0, 0]
                        selected = [0, 0, 0]
                        for i in range(sizes[0]):
                             imgH = data_stripped[i:i + 1, :, :]
                             if entropy(imgH) > max_entropy[0]:
                                 selected[0] = i
                                 max_entropy[0] = entropy(imgH)

                        for i in range(sizes[1]):
                            imgC = data_stripped[:, i:i + 1, :]
                            if entropy(imgC) > max_entropy[1]:
                                selected[1] = i
                                max_entropy[1] = entropy(imgC)

                        for i in range(sizes[2]):
                            imgS = data_stripped[:, :, i:i + 1]
                            if entropy(imgS) > max_entropy[2]:
                                selected[2] = i
                                max_entropy[2] = entropy(imgS)

                        imageH =    data_stripped[selected[0]:selected[0] + 1, :, :][0]
                        imageC =    data_stripped[:, selected[1]:selected[1] + 1, :][:,0,:]
                        imageS =    data_stripped[:, :, selected[2]:selected[2] + 1]

                        nb.save(nb.Nifti1Image(imageH,img.affine),new_path_data + "_st_H.nii")
                        nb.save(nb.Nifti1Image(imageS, img.affine), new_path_data + "_st_S.nii")
                        nb.save(nb.Nifti1Image(imageC, img.affine), new_path_data + "_st_C.nii")
                    except:
                        print(filename)



def embbededName():
    for dirname, dirnames, filenames in os.walk(PATH_DATA):
        for filename in filenames:
            if '.nii' in filename and '___' not in filename:
                path_data = os.path.join(dirname, filename)
                # Nos quedamos con las imágenes del centro.
                patienID, imgID, fecha = getInformation(filename)
                for fi in csv_files:
                    if (fi[0] != '.'):
                        df = pd.read_csv(os.path.join(PATH_INFORMATION, fi))
                        res = [i for i in df.columns if 'Diagnosis' in i]
                        if (df[df['Image.ID'] == int(imgID)].__len__() > 0):
                            diag = df[df['Image.ID'] == int(imgID)][res].values[0][0]
                            newname = filename[:filename.find('.nii')] + '___' + diag + '.nii'
                            os.renames(path_data, os.path.join(dirname, newname))
                            break
embbededName()