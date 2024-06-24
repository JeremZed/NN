import nnz.tools as tools
import nnz.config as config
from nnz.project.project import Project
import os
import numpy as np
import pandas as pd
import shutil
import cv2
import h5py
import io
from PIL import Image
import matplotlib.pyplot as plt
import time
from memory_profiler import profile
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False

from keras.src.legacy.preprocessing.image import ImageDataGenerator

class ProjectNN(Project):
    def __init__(self, name=None):
        super().__init__(name=name)

class ProjectNNClassifier(ProjectNN):

    def __init__(self, name=None):
        super().__init__(name=name)

    def split_images_to_sets_folders(self, path_src, path_dst, size_train=0.6, size_val=0.15, size_test=0.25, limit=-1, random_state=123, verbose=0, shuffle=True, resize=None, augmentation_data = None, nb_augmentation = 0 ):
        """ Permet de répartir le contenu d'un dossier contenant plusieurs classes d'images dans les dossiers train, validation, test avec un ratio
            Le contenu du dossier d'origine doit être composé comme suit :
                class_1/
                    img_1.jpg
                    img_2.jpg
                    ...
                class_2/
                    img_1.jpg
                    img_2.jpg
                    ...
                ...
        """

        if os.path.exists(path_src) != True:
            raise Exception(f"Dossier source inconnu : {path_src}")
        if os.path.exists(path_dst) != True:
            raise Exception(f"Dossier de destination inconnu : {path_dst}")
        if size_train + size_val + size_test != 1.0:
                raise ValueError( f'Le cumul de size_train:{size_train}, size_val:{size_val}, size_test:{size_test} n\'est pas égal à 1.0')

        # Création des dossiers de destination

        last_char = path_dst[-1]
        if last_char != "/":
            path_dst = path_dst + "/"

        train_path = path_dst + "train"
        val_path = path_dst + "val"
        test_path = path_dst + "test"

        tools.create_directory(train_path)
        tools.create_directory(val_path)
        tools.create_directory(test_path)

        np.random.seed(random_state)

        classes_directories = tools.list_dirs(path_src)

        # On récupère le nombre minimum d'images présent dans les classes
        count_image_min = 9999999999999
        for i, c in enumerate(classes_directories):
            class_name = c.split("/")[-1]
            files = tools.list_files(c)
            images_count = len(files)

            if images_count < count_image_min:
                count_image_min = images_count

        limit_images = count_image_min if limit == -1 else limit

        classes = []

        dataset_train = []
        dataset_val = []
        dataset_test = []

        for i, c in enumerate(classes_directories):
            class_name = c.split("/")[-1]

            files = tools.list_files(c)

            if limit > -1 and limit_images < len(files):
                files = files[:limit_images]

            if shuffle == True:
                np.random.shuffle(files)

            total_size = len(files)

            train_files, val_files, test_files = np.split(
                files,
                [
                    int(total_size*size_train),
                    int(total_size*(size_train + size_val))
                ])

            # Création des dossiers pour chaque classe
            train_class_path = train_path+"/"+class_name
            val_class_path = val_path+"/"+class_name
            test_class_path = test_path+"/"+class_name

            # On supprime l'existant
            tools.remove_directory(train_class_path)
            tools.remove_directory(val_class_path)
            tools.remove_directory(test_class_path)

            tools.create_directory(train_class_path)
            tools.create_directory(val_class_path)
            tools.create_directory(test_class_path)

            o = {
                "name" : class_name,
                "indice" : i,
                "files" : {
                    "train" : train_files,
                    "val" : val_files,
                    "test" : test_files
                },
                "paths" : {
                    "train" : train_class_path,
                    "val" : val_class_path,
                    "test" : test_class_path
                },
                "meta" : {
                    "count_train" : len(train_files),
                    "count_val" : len(val_files),
                    "count_test" : len(test_files),
                }
            }
            classes.append(o)

            if verbose > 0:
                print(class_name, len(files), "fichiers")
                print("size_train : " , o['meta']['count_train'], f"({size_train})")
                print("size_val : ",  o['meta']['count_val'], f"({size_val})")
                print("size_test : ", o['meta']['count_test'], f"({size_test})")
                print("size_total : ", (o['meta']['count_train'] +  o['meta']['count_val'] + o['meta']['count_test']), f"({(size_train + size_val + size_test)})")

        images_train = []
        images_val = []
        images_test = []

        if augmentation_data is not None:
            datagen = ImageDataGenerator(**augmentation_data)
            nb_augmentation = 1 if nb_augmentation == 0 else nb_augmentation
        else:
            datagen = None

        # Copie des images et déplacement dans les bons répertoire
        for i, c in enumerate(classes):
            for f in c['files']['train']:

                img = cv2.imread(f)
                if resize is not None:
                    img = cv2.resize(img, resize )
                img_dest = f'{c['paths']['train']}/{f.split("/")[-1]}'
                cv2.imwrite(img_dest, img)

                images_train.append({ "class" : c["indice"], "path" : img_dest})

                # Images augmentées
                if datagen is not None:
                    _x = img.reshape((1,) + img.shape)
                    t = datagen.flow(_x, batch_size=1)
                    _i = 0
                    img_d_dest = f'{c['paths']['train']}/d-{_i}-{f.split("/")[-1]}'
                    for batch in t:
                        # dataset_train.append((batch[0], i))
                        cv2.imwrite(img_d_dest, batch[0])
                        images_train.append({ "class" : c["indice"], "path" : img_d_dest})
                        _i+=1
                        if _i % nb_augmentation == 0:
                            break

            for f in c['files']['val']:
                img = cv2.imread(f)
                if resize is not None:
                    img = cv2.resize(img, resize )
                img_dest = f'{c['paths']['val']}/{f.split("/")[-1]}'
                cv2.imwrite(img_dest, img)
                images_val.append({ "class" : c["indice"], "path" : img_dest})

            for f in c['files']['test']:
                img = cv2.imread(f)
                if resize is not None:
                    img = cv2.resize(img, resize )
                img_dest = f'{c['paths']['test']}/{f.split("/")[-1]}'
                cv2.imwrite(img_dest, img)
                images_test.append({ "class" : c["indice"], "path" : img_dest})

        np.random.shuffle(images_train)
        np.random.shuffle(images_val)
        np.random.shuffle(images_test)

        moment = f"{tools.get_format_date(pattern="%Y_%m_%d_%H_%M_%S")}"
        self.create_h5_dataset_img('train', images_train, to=moment)
        self.create_h5_dataset_img('val', images_val, to=moment)
        self.create_h5_dataset_img('test', images_test, to=moment)

        path_dir_moment = f"{config.__path_dir_datasets__}/{moment}/"
        datasets = self.read_h5_datasets(path_dir=path_dir_moment)

        return datasets, path_dir_moment


    def create_h5_dataset_img(self, name, datas, to=None):
        """ Permet de créer et de le remplir itérativement """

        if to is None:
            to = f"{tools.get_format_date(pattern="%Y_%m_%d_%H_%M_%S")}"

        # On récupère la première image pour récupérer les caractéristiques
        first_image_path = datas[0]['path']

        img_array = tools.convert_img_to_pixel(first_image_path)

        shape = img_array.shape
        first_img = img_array.reshape( (1,) + shape)

        dir_path = f'{config.__path_dir_datasets__}/{to}'
        h5_dataset_X_path = f'{dir_path}/img_X_{name}.h5'
        h5_dataset_y_path = f'{dir_path}/img_y_{name}.h5'

        tools.create_directory(dir_path)

        if os.path.exists(h5_dataset_X_path):
            os.remove(h5_dataset_X_path)
        if os.path.exists(h5_dataset_y_path):
            os.remove(h5_dataset_y_path)

        # On peut créer le dataset h5 au format de l'image
        hf_x = h5py.File(h5_dataset_X_path, 'w')
        h5_dataset_x = hf_x.create_dataset(f'X_{name}', first_img.shape, data=first_img ,chunks=True, maxshape=tuple([None]*len(first_img.shape)))

        # Création du dataset h5 de la target
        hf_y = h5py.File(h5_dataset_y_path, 'w')
        h5_dataset_y = hf_y.create_dataset(f'y_{name}', (1,), data=[datas[0]['class']] ,chunks=True, maxshape=(None,))

        # On récupère ensuite le dataset en mode append pour remplir le dataset des pixels/bytes de la chaque image
        hf_x = h5py.File(h5_dataset_X_path, 'a')
        h5_dataset_x = hf_x.get(f'X_{name}')
        # Idem pour la target de chaque image
        hf_y = h5py.File(h5_dataset_y_path, 'a')
        h5_dataset_y = hf_y.get(f'y_{name}')

        # On parcours la liste des chemins des images à partir de la deuxième image, vu que la première à déjà été mise dans le dataset lors de sa création
        for img in datas[1:]:

            # On ajoute un nouvel enregistrement au fur et à mesure
            h5_dataset_x.resize( (h5_dataset_x.shape[0] + 1, ) + shape)
            h5_dataset_y.resize( (h5_dataset_y.shape[0] + 1, ))

            img_array = tools.convert_img_to_pixel(img['path'])
            h5_dataset_x[h5_dataset_x.shape[0]-1] = img_array

            # La target
            h5_dataset_y[h5_dataset_y.shape[0]-1] = img['class']

        return None

    def read_h5_datasets(self, path_dir=None):
        """ Permet de retourner le contenu des datasets enregistrés au format h5 """

        files = tools.list_files(path_dir)

        X_train = np.array([])
        y_train = np.array([])
        X_val = np.array([])
        y_val = np.array([])
        X_test = np.array([])
        y_test = np.array([])

        for file in files:
            if "X_train" in file:
                X_train = np.array(tools.h5_read(file, "X_train"))
            if "X_val" in file:
                X_val = np.array(tools.h5_read(file, "X_val"))
            if "X_test" in file:
                X_test = np.array(tools.h5_read(file, "X_test"))
            if "y_train" in file:
                y_train = np.array(tools.h5_read(file, "y_train"))
            if "y_val" in file:
                y_val = np.array(tools.h5_read(file,"y_val"))
            if "y_test" in file:
                y_test = np.array(tools.h5_read(file, "y_test"))

        return X_train, y_train, X_val, y_val, X_test, y_test


