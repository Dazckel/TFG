import os
import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
from itertools import combinations
from sklearn.utils import resample
from pathlib import Path
import nibabel as nb
import numpy as np
import re
import platform
from numpy.random import default_rng



PATH_ROOT = Path(os.path.dirname(__file__)).parent
Diagnosis = ["NL", "MCI", "AD"]
PATH_DATASET = ""
operating_system = platform.system()

if operating_system == 'Windows':
    PATH_ADNI_IMAGES = Path('F:/') / 'Dataset/ADNI/NewImages'
    PATH_INFORMATION = Path('F:/') / '/Dataset/ADNI/Informacion/DatosImagenes'
    csv_file = PATH_INFORMATION / "super_dataframe.csv"
elif operating_system == 'Linux':
    PATH_ADNI_IMAGES = PATH_ROOT / 'Datos' / 'Dataset' / 'ADNI' / 'NewImages'
    PATH_INFORMATION = Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/ADNI/Informacion/DatosImagenes'
    csv_file = PATH_INFORMATION / "super_dataframe.csv"

class SiameseNetworkDataset(Dataset):
    def __init__(self, files_and_clases=None, imageFolderDataset=PATH_DATASET, path_root=PATH_ROOT, transform=None,
                 dataset_csv=None):

        # El dataset se crea pasándole las imágenes que le pertenecen, junto a sus etiquetas
        if not (files_and_clases == None):
            self.files = files_and_clases[0]
            self.classes = files_and_clases[1]

        self.imageFolderDataset = imageFolderDataset  # Ruta de la carpeta que contiene las imágenes.
        self.transform = transform  # Transformaciones realizadas sobre las imágenes

        self.path_root = path_root  # Ruta archivos entrenamiento

        # En caso de que indiquemos que NO indiquemos un archivo CSV,
        # crear los pares de  imágenes que alimentarán la red.
        if dataset_csv == None:
            file_pairs, lbs = self.createSamples()

            self.file_names = file_pairs  # Pares de imágenes equitativamente distribuidos
            self.labels = lbs  # Etiquetas mostrando
        else:  # En caso de que hayamos creado ya el dataset, leemos los pares de imágenes y la etiqueta asociada
            self.path_dataset = dataset_csv
            df = pd.read_csv(dataset_csv)
            self.file_names = list(zip(df.image1, df.image2))
            self.labels = df.Y

    # Crea los items de nuestro dataset.
    def createSamples(self):

        # Creamos una lista de combinaciones entre imágenes
        filenames_classes = dict(zip(self.files, self.classes))  # Pares archivo-clase
        file_pairs = list(combinations(self.files, 2))  # Creamos pares de archivos

        # Como las combinacioens aparecerán repetidas {(a,b) == (b,a)}
        # convertimos la lista en un conjunto para eliminarlas
        lbs = []
        x0 = []
        x1 = []
        # Iteramos sobre los pares para conocer sus clases
        for filename1, filename2 in file_pairs:
            class1 = filenames_classes[filename1]
            class2 = filenames_classes[filename2]
            if class1 == class2:
                lbs.append(0)  # Ambas imágenes pertenecen a la misma clase.
            else:
                lbs.append(1)  # Las imágenes no pertenecen a la misma clase.
            x0.append(filename1)
            x1.append(filename2)

        # Creamos un dataframe de pandas
        final_dataset = pd.DataFrame(data=zip(x0, x1, lbs), columns=['x0', 'x1', 'Y'])

        # Balanceamos el dataset, ya que habrá más instancias de una clase que de otra
        # en concreto hay más imágenes que no coinciden que que si coinciden.
        # Es importante balancear el dataset para evitar sesgar el modelo
        df1 = final_dataset[final_dataset.Y == 0]
        df2 = final_dataset[final_dataset.Y == 1]
        samples = min([len(df1), len(df2)])
        df1_balanced = resample(df1, replace=False, n_samples=samples, random_state=2)
        df2_balanced = resample(df2, replace=False, n_samples=samples, random_state=2)

        final_dataset = pd.concat([df1_balanced, df2_balanced]).reset_index()
        file_pairs = list(zip(final_dataset.x0, final_dataset.x1))
        lbs = list(final_dataset.Y)

        # Devolvemos la lista de imágenes y etiquetas.
        return file_pairs, lbs


    def getTriplet(self,idx):
        img1 = self.file_names[idx][0]
        img2 = self.file_names[idx][1]
        label = self.labels[idx]

        return img1, img2, label
    # Función heredada que devuelve la instancia de índie idx del dataset.
    def __getitem__(self, idx):
        img1 = self.file_names[idx][0]
        img2 = self.file_names[idx][1]

        img1 = nb.load(os.path.join(self.imageFolderDataset, img1)).get_fdata()
        img2 = nb.load(os.path.join(self.imageFolderDataset, img2)).get_fdata()
        label = self.labels[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    # Nos devuelce cuantas imágenes de cada clase tenemos.
    def getStatistic(self):
        total_img = len(os.listdir(self.imageFolderDataset))
        print(f'NL: {len(self.NL)}: {len(self.NL) / total_img * 100}%')
        print(f'MCI: {len(self.MCI)}: {len(self.MCI) / total_img * 100}%')
        print(f'AD: {len(self.AD)}: {len(self.AD) / total_img * 100}%')

    def __len__(self):
        return len(self.file_names)


# TRANSFORMACIONES SOBRE LOS DATOS

# Rescalado y selección de slice
class Rescale(object):
    def __init__(self, output=100):
        assert isinstance(output, (int, tuple))
        self.output_size = output

    def __call__(self, image):
        resize = torchvision.transforms.Resize((self.output_size, self.output_size), antialias=True)
        return resize(image[None,:,:])


class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image)


def save_to_csv(subset, path):
    size = subset.__len__()
    labels = [None] * size
    imgs1 = [None] * size
    imgs2 = [None] * size

    for i in range(0,size):
        img1,img2,label = subset.getTriplet(i)
        labels[i] = label
        imgs1[i] = img1
        imgs2[i] = img2

    new_df = pd.DataFrame(data=zip(imgs1, imgs2, labels), columns=["image1", "image2", "Y"])
    new_df.to_csv(path)


def createDataset(path_dataset):
    seed = 42
    np.random.seed(seed)

    NL = []
    MCI = []
    AD = []
    df = pd.read_csv(PATH_INFORMATION / "super_dataframe.csv")
    df1 = pd.read_csv(os.path.join(Path('F:/') / '/Dataset/ADNI/Informacion/DXSUM_PDXCONV_ADNIALL_23Nov2023.csv'))
# Selección de imágenes T1 con mismo preprocesado.
    images = getADNIfiles()
    imgs   = getCoronal(images[0])
    filenames = getSamePreproADNI(imgs)[2]
    counter = 0
    for filename in filenames:
        diag= getDiagnosis(filename,df)
        diagAux = getDiagnosisAux(filename,df1,df)

        #Solo cojemos los archivos donde haya coincidencia de diagnóstico.
        if (diagAux == diag):
            if Diagnosis[0] == diag:
                NL.append(filename)
            elif Diagnosis[1] == diag:
                MCI.append(filename)
            elif Diagnosis[2] == diag:
                AD.append(filename)
        else:
            counter += 1

    number_NL = len(NL)
    number_MCI = len(MCI)
    number_AD = len(AD)

    # Averiguamos de que clase tenemos menos instancias.
    minNInstances = np.min([number_NL, number_MCI, number_AD])
    # ahora nos quedamos en cada clase con ese número de instancias
    NL = NL[:minNInstances]
    MCI = MCI[:minNInstances]
    AD = AD[:minNInstances]

    size_database = minNInstances*3

    filenames = []
    filenames.extend(NL)
    filenames.extend(MCI)
    filenames.extend(AD)

    classes = [Diagnosis[0]]*len(NL) + [Diagnosis[1]]*len(MCI) + [Diagnosis[2]]*len(AD)

    # Bien, debido que una vez echas las combinaciones nos sería muy tedioso
    # evitar datasnooping ( ya que nuestros items consistirían en parejas de imágenes
    # y averiguar si una imagen esta en alguna pareja de X dataset es difícil e inapropiado)
    # realizamos el split ahora.

    nTest = int(0.1 * size_database)
    nTrain = int(0.8 * size_database * 0.9)
    nVad = int(0.2 * size_database * 0.9)
    rng = default_rng()
    # Generamos índices aleatorios para obtener los archivos de cada dataset
    random_idxs = rng.choice( len(filenames),size_database, replace=False).tolist()

    files_Train = [filenames[random_idxs[i]] for i in range(0, nTrain)]
    class_Train = [classes[random_idxs[i]] for i in range(0, nTrain)]

    files_Vad = [filenames[random_idxs[i]] for i in range(nTrain, nVad + nTrain)]
    class_Vad = [classes[random_idxs[i]] for i in range(nTrain, nVad + nTrain)]

    files_Test = [filenames[random_idxs[i]] for i in range(nTrain + nVad, nTrain + nVad + nTest)]
    class_Test = [classes[random_idxs[i]] for i in range(nTrain + nVad, nTrain + nVad + nTest)]

    #  Ahora ya tenemos los archivos que vamos a dedicar a cada clase
    train_dataset = SiameseNetworkDataset(imageFolderDataset=PATH_DATASET, files_and_clases=[files_Train, class_Train])
    valid_dataset = SiameseNetworkDataset(imageFolderDataset=PATH_DATASET, files_and_clases=[files_Vad, class_Vad])
    test_dataset = SiameseNetworkDataset(imageFolderDataset=PATH_DATASET, files_and_clases=[files_Test, class_Test])

    # Ahora debemos evitar el data snooping, ya que muchas de las imágenes en un conjunto, estarán el otros
    # Guardamos los diferentes datasets:

    save_to_csv(train_dataset, os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV', 'train_all_classes_T1_same.csv'))
    save_to_csv(valid_dataset, os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV', 'valid_all_classes_T1_same.csv'))
    save_to_csv(test_dataset, os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV', 'test_all_classes_T1_same.csv'))

    print("Train examples: ", train_dataset.__len__())
    print("Valid examples: ", valid_dataset.__len__())
    print("Test examples: ", test_dataset.__len__())


def strenght(file):
    if '3T' in file:
        return '3T'
    elif '1.5T' in file:
        return '1.5T'
    else:
        return '0'

def getADNIfiles():
    ####################
    img_names_T1 = set()
    img_names_T3 = set()
    filenames_T3 = set()
    filenames_T1 = set()
    ####################
    for dirname, dirnames, filenames in os.walk(PATH_ADNI_IMAGES):
        for filename in filenames:
            if 'nii' in filename:
                filename = eraseRedundancy(filename)
                len_T1 = filenames_T1.__len__()
                len_T3 = filenames_T3.__len__()
                ####################
                if strenght(dirname) == '1.5T':
                    filenames_T1.add(filename)
                    if(len_T1 != filenames_T1.__len__()):
                        filename = restoreRedundancy(filename,dirname)
                        img_names_T1.add(dirname + '\\' + filename)
                ####################
                elif strenght(dirname) == '3T':
                    filenames_T3.add(filename)
                    if(len_T3 != filenames_T3.__len__()):
                        filename = restoreRedundancy(filename, dirname)
                        img_names_T3.add(dirname + '\\' + filename)
                ####################

    return [img_names_T1, img_names_T3]
def getNumberOperation(fileName):
    counter = 0
    if 'N3' in fileName:
        counter += 1
    if 'B1' in fileName:
        counter += 1
    if 'GradWarp' in fileName:
        counter += 1
    return counter


def getSamePreproADNI(images):
    op_1 = []
    op_2 = []
    op_3 = []
    for image in images:
        if getNumberOperation(image) == 1:
            op_1.append(image)
        elif getNumberOperation(image) == 2:
            op_2.append(image)
        elif getNumberOperation(image) == 3:
            op_3.append(image)

    return [op_1, op_2, op_3]

def getCoronal(files):
    coronal_files = []
    for file in files:
        if '_st_C' in file:
            coronal_files.append(file)
    return coronal_files

def getSagital(files):
    sagital_files = []
    for file in files:
        if '_st_S' in file:
            sagital_files.append(file)

    return sagital_files

def getHorizontal(files):
    horizontal_files = []
    for file in files:
        if '_st_H' in file:
            horizontal_files.append(file)

    return horizontal_files
def getInformation(fn):
    filename = fn[::-1][:fn[::-1].find('\\')][::-1]
    d = filename.find('S')
    patienID = filename[d - 4:d + 6]
    d = filename.find('_I')
    p = filename.find('.')
    imgID = filename[d + 1:p]
    imgID = re.sub("[^0-9]", "", imgID)
    d = filename.find('Br_') + 3
    fecha = (fn[:fn.find('\\I')][::-1])[:fn[:fn.find('\\I')][::-1].find('\\')][::-1][:(fn[:fn.find('\\I')][::-1])[:fn[:fn.find('\\I')][::-1].find('\\')][::-1].find('_')]
    return [patienID, imgID, fecha]


def getDiagnosis(filename,df):
    diag = ""
    patienID, imgID, fecha = getInformation(filename)
    fecha = toDateAux(fecha)
    res = [i for i in df.columns if 'Diagnosis' in i]
    if (df[(df['Image.ID'] == int(imgID)) & (df['PTID'] == patienID)& (df['Scan.Date'] == fecha)].__len__() > 0):
        diag = df[(df['Image.ID'] == int(imgID)) & (df['PTID'] == patienID) & (df['Scan.Date'] == fecha)][res].values[0][0]
        return diag
    return diag

def toDateAux(date):
    date[5:7] + '/' + date[8:] + '/' + date[:4]
    day = date[8:]
    month = date[5:7]
    year = date[2:4]

    if day[0] == '0':
        day = day[1]
    if month[0] == '0':
        month = month[1]
    return month + '/' + day + '/' + year

def getInfoDiag(filename,df):
    patienID, imgID, fecha = getInformation(filename)
    fecha = toDateAux(fecha)
    if (df[(df['Image.ID'] == int(imgID)) & (df['PTID'] == patienID) & (df['Scan.Date'] == fecha)].__len__() > 0):
        rid = df[(df['Image.ID'] == int(imgID)) & (df['PTID'] == patienID) & (df['Scan.Date'] == fecha)]['RID'].values[0]
        viscode = df[(df['Image.ID'] == int(imgID)) & (df['PTID'] == patienID) & (df['Scan.Date'] == fecha)]['Visit'].values[0]
        if viscode == 'Month 12':
            viscode = "m12"
        elif viscode == 'Month 6':
            viscode = "m06"
        elif viscode == 'Month 24':
            viscode = "m24"
        elif viscode == 'Month 18':
            viscode = "m18"
        elif viscode == 'Month 36':
            viscode = "m36"
        elif viscode == 'Month 48':
            viscode = "m48"
        elif viscode == 'Screening':
            viscode = "bl"
        elif viscode == 'Baseline':
            viscode = "bl"
        return rid,viscode
def getDiagnosisAux(filename,df,df2):
    patienID, imgID, fecha = getInformation(filename)
    try:
        rid, viscode = getInfoDiag(filename,df2)
    except Exception :
        rid = 231
        viscode = 'bb'

    df = df[df['Phase'] == 'ADNI1']
    diag = df[(df['RID'] == rid) & (df['PTID'] == patienID) & (df['VISCODE'] == viscode)]['DXCURREN']
    if diag.empty:
        return ""
    return Diagnosis[int(diag) - 1]


def eraseRedundancy(filename):
    if 'Scaled_2' in filename:
        filename = filename.replace('Scaled_2', 'Scaled')
    if 'MPR-R' in filename:
        filename = filename.replace('MPR-R', 'MPR')

    return filename

def restoreRedundancy(filename,dirname):
    if 'MPR-R' in dirname:
        filename = filename.replace('MPR', 'MPR-R')
    if 'Scaled_2' in dirname:
        filename = filename.replace('Scaled', 'Scaled_2')

    return filename


# for i in range(0,size):
#     i1,i2,l = train_dataset.getTriplet(i)
#     if (getDiagnosis(i1,df) == getDiagnosis(i2,df)) and l != 0:
#         print("MIERDA")
#     if (getDiagnosis(i1,df) != getDiagnosis(i2,df)) and l != 1:
#         print("MIERDA")

def norm_st(images_1,res,batch_size):
    images_11 = images_1[:,None,:,:].clone()
    images_11 = images_11.view(images_1.size(0), -1)


    #Standardization
    means = images_11.mean(1, keepdim=True)
    stds = images_11.std(1, keepdim=True)
    images_11 -= means
    images_11 /=stds
    #Normalization
    # mins=images_11.min(1, keepdim=True)[0]
    # maxs = images_11.max(1, keepdim=True)[0]
    # images_11 -= mins
    # images_11 /= (maxs-mins)


    return images_11.view(batch_size, res, res)[:,None,:,:]
