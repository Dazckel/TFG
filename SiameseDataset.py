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
from skimage.measure import shannon_entropy as entropy

PATH_ROOT = Path(os.path.dirname(__file__)).parent
PATH_ADNI_IMAGES = PATH_ROOT / 'Datos' / 'Dataset' / 'ADNI' / 'NewImages'
PATH_DATASET = ""
Diagnosis = ["NL", "MCI", "AD"]


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
        self.transform = transform  # Transformaciones

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
        file_pairs = list(set(file_pairs))

        # Creamos la lista de etiquetas para cada par de imágenes
        # que son los items de nuestra base de datos
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
        df1_balanced = resample(df1, replace=True, n_samples=samples, random_state=2)
        df2_balanced = resample(df2, replace=True, n_samples=samples, random_state=2)

        final_dataset = pd.concat([df1_balanced, df2_balanced]).reset_index()
        file_pairs = list(zip(final_dataset.x0, final_dataset.x1))
        lbs = list(final_dataset.Y)

        # Devolvemos la lista de imágenes y etiquetas.
        return file_pairs, lbs

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

        return img1, img2, label, self.file_names[idx][0], self.file_names[idx][1]

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

        sizes = [image.shape[0], image.shape[1], image.shape[2]]
        max_entropy = [0, 0, 0]
        selected = [0, 0, 0]

        # for i in range(sizes[0]):
        #     imgH = image[i:i + 1, :, :]
        #     if entropy(imgH) > max_entropy[0]:
        #         selected[0] = i
        #         max_entropy[0] = entropy(imgH)

        for i in range(sizes[1]):
            imgC = image[:, i:i + 1, :]
            if entropy(imgC) > max_entropy[1]:
                selected[1] = i
                max_entropy[1] = entropy(imgC)

        # for i in range(sizes[2]):
        #     imgS = image[:, :, i:i + 1]
        #     if entropy(imgS) > max_entropy[2]:
        #         selected[2] = i
        #         max_entropy[2] = entropy(imgS)

        # imageH = image[:, :, selected[2]:selected[2] + 1][:, :0]
        imageC = image[:, selected[1]:selected[1] + 1, :][:, 0, :]
        # imageS = image[selected[0]:selected[0] + 1, :, :][0, :, :]

        return resize(imageC[None, :, :])


class toSlice(object):
    def __call__(self, image):
        sizes = [image.shape[0], image.shape[1], image.shape[2]]
        max_entropy = [0, 0, 0]
        selected = [0, 0, 0]

        for i in range(sizes[0]):
            imgH = image[i:i + 1, :, :]
            if entropy(imgH) > max_entropy[0]:
                selected[0] = i
                max_entropy[0] = entropy(imgH)

        for i in range(sizes[1]):
            imgC = image[:, i:i + 1, :]
            if entropy(imgC) > max_entropy[1]:
                selected[1] = i
                max_entropy[1] = entropy(imgC)

        for i in range(sizes[2]):
            imgS = image[:, :, i:i + 1]
            if entropy(imgS) > max_entropy[2]:
                selected[2] = i
                max_entropy[2] = entropy(imgS)

        imageH = image[:, :, selected[2]:selected[2] + 1]
        imageC = image[:, selected[1]:selected[1] + 1, :]
        imageS = image[selected[0]:selected[0] + 1, :, :]

        return imageC


class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image)


def save_to_csv(subset, path):
    labels = [None] * subset.__len__()
    imgs1 = [None] * subset.__len__()
    imgs2 = [None] * subset.__len__()
    for i, (_, _, label, img1, img2) in enumerate(subset):
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
    for img in os.listdir(path_dataset):
        if Diagnosis[0] in img:
            NL.append(img)
        elif Diagnosis[1] in img:
            MCI.append(img)
        elif Diagnosis[2] in img:
            AD.append(img)

    number_NL = len(NL)
    number_MCI = len(MCI)
    number_AD = len(AD)

    # Averiguamos de que clase tenemos menos instancias.
    minNInstances = np.min([number_NL, number_MCI, number_AD])
    # ahora nos quedamos en cada clase con ese número de instancias
    NL = NL[:minNInstances]
    MCI = MCI[:minNInstances]
    AD = AD[:minNInstances]

    filenames = []
    filenames.extend(NL)
    filenames.extend(MCI)
    filenames.extend(AD)

    classes = []
    for i in filenames:
        if i in NL:
            classes.append(Diagnosis[0])
        elif i in MCI:
            classes.append(Diagnosis[1])
        elif i in AD:
            classes.append(Diagnosis[2])

    # Bien, debido que una vez echas las combinaciones nos sería muy tedioso
    # evitar datasnooping ( ya que nuestros items consistirían en parejas de imágenes
    # y averiguar si una imagen esta en alguna pareja de X dataset es difícil e inapropiado)
    # realizamos el split ahora.

    nTest = int(0.2 * minNInstances)
    nTrain = int(0.8 * minNInstances * 0.8)
    nVad = int(0.2 * minNInstances * 0.8)

    # Generamos índices aleatorios para obtener los archivos de cada dataset
    random_idxs = np.random.randint(0, len(filenames), size=minNInstances).tolist()

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

    save_to_csv(train_dataset, os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV', 'train_all_classes.csv'))
    save_to_csv(valid_dataset, os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV', 'valid_all_classes.csv'))
    save_to_csv(test_dataset, os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV', 'test_all_classes.csv'))

    print("Train examples: ", train_dataset.__len__())
    print("Valid examples: ", valid_dataset.__len__())
    print("Test examples: ", test_dataset.__len__())


def isT1(file):
    if '3T' in file:
        return '3T'
    elif '1.5T' in file:
        return '1.5T'
    else:
        return '0'


def getADNIfiles():
    img_names_T1 = []
    img_names_T3 = []
    for dirname, dirnames, filenames in os.walk(PATH_ADNI_IMAGES):
        for filename in filenames:
            if isT1(dirname) == '1.5T':
                img_names_T1.append(filename)
            elif isT1(dirname) == '3T':
                img_names_T3.append(filename)
    return [set(img_names_T1), set(img_names_T3)]


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


images = getADNIfiles()
samePrePro_T1 = getSamePreproADNI(images[0])
samePrePro_T3 = getSamePreproADNI(images[1])

msg_T1 = f'Imágenes tomadas con escáneres T1: \n\t -Procesado de 1 sola operación:\n\t\t {samePrePro_T1[0]}' + \
         f'\n\t -Procesado de 2 operaciones:\n\t\t {samePrePro_T1[1]}' \
         f'\n\t -Procesado de 3 operaciones:\n\t\t {samePrePro_T1[2]}'

msg_T3 = f'Imágenes tomadas con escáneres T1: \n\t -Procesado de 1 sola operación:\n\t\t {samePrePro_T3[0]}' + \
         f'\n\t -Procesado de 2 operaciones:\n\t\t {samePrePro_T3[1]}' \
         f'\n\t -Procesado de 3 operaciones:\n\t\t {samePrePro_T3[2]}'
