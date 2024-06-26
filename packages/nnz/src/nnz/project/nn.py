import nnz.tools as tools
import nnz.config as config
from nnz.project.project import Project
from nnz.augmentation.image import ImageAugmentation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import keras
import shutil
import cv2
import h5py
import io
# from PIL import Image
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import time
from memory_profiler import profile

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = False

from keras.src.legacy.preprocessing.image import ImageDataGenerator

class ProjectNN(Project):
    def __init__(self, name=None):
        super().__init__(name=name)

    def fit_model(self, model, X, y, X_val, y_val, name='model', batch_size=16, epochs=1, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1 ):
        """ Permet de lancer le fit du modèle avec les différents callback de sauvegarde du modèle """

        checkpoint_filepath = f'./runtime/ckpt/checkpoint.{name}.keras'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only)

        csv_logger = keras.callbacks.CSVLogger(f'./runtime/{name}.history.log')

        tb_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)

        with open(f'./runtime/{name}.summary.json', "w") as json_file:
            json_file.write(model.to_json())

        return model.fit(  X, y,
                      batch_size      = batch_size,
                      epochs          = epochs,
                      verbose         = verbose,
                      validation_data = (X_val, y_val),
                      callbacks=[model_checkpoint_callback, csv_logger, tb_callback])

    def load_model(self, name):
        """ Permet de charger un modèle """

        checkpoint_filepath = f'./runtime/ckpt/checkpoint.{name}.keras'
        return keras.models.load_model(checkpoint_filepath)

    def model_get_predict_max(self, model, datas, verbose=1):
        """ Permet de retourner les prédictions """
        predictions = model.predict(datas, verbose=verbose)
        return np.argmax(predictions, axis=-1)

    def load_summary_model(self, name):
        """ Permet de retourner l'architecture d'un modèle """

        f = open(f'./runtime/{name}.summary.json', 'r')
        config = f.read()
        f.close()
        model = keras.models.model_from_json(config)

        return model.summary()

class ProjectNNClassifier(ProjectNN):

    def __init__(self, name=None):
        super().__init__(name=name)

        self.__classes = None

    def get_classes(self):
        """ Permet de retourner la liste des classes """
        return self.__classes

    def split_images_to_sets_folders(self, path_src, path_dst, size_train=0.6, size_val=0.15, size_test=0.25, limit=-1, random_state=123, verbose=0, shuffle=True, resize=None, augmentation_data = None):
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
            datagen = ImageAugmentation(**augmentation_data)
        else:
            datagen = None

        # Copie des images et déplacement dans les bons répertoire
        for i, c in enumerate(classes):

            count_files_train = 0
            count_files_val = 0
            count_files_test = 0

            for f in c['files']['train']:

                img = cv2.imread(f)
                if resize is not None:
                    img = cv2.resize(img, resize )
                img_dest = f'{c['paths']['train']}/{f.split("/")[-1]}'
                cv2.imwrite(img_dest, img)

                images_train.append({ "class" : c["indice"], "path" : img_dest})
                count_files_train+=1

                # Images augmentées
                if datagen is not None:

                    new_images = datagen.generate(img, type="array")
                    for _i, ni in enumerate(new_images):
                        img_d_dest = f'{c['paths']['train']}/d-{_i}-{f.split("/")[-1]}'
                        cv2.imwrite(img_d_dest, ni)
                        images_train.append({ "class" : c["indice"], "path" : img_d_dest})
                        count_files_train+=1

            for f in c['files']['val']:
                img = cv2.imread(f)
                if resize is not None:
                    img = cv2.resize(img, resize )
                img_dest = f'{c['paths']['val']}/{f.split("/")[-1]}'
                cv2.imwrite(img_dest, img)
                images_val.append({ "class" : c["indice"], "path" : img_dest})
                count_files_val+=1

            for f in c['files']['test']:
                img = cv2.imread(f)
                if resize is not None:
                    img = cv2.resize(img, resize )
                img_dest = f'{c['paths']['test']}/{f.split("/")[-1]}'
                cv2.imwrite(img_dest, img)
                images_test.append({ "class" : c["indice"], "path" : img_dest})
                count_files_test+=1

            classes[i]['meta']['count_train'] = count_files_train
            classes[i]['meta']['count_val'] = count_files_val
            classes[i]['meta']['count_test'] = count_files_test

        class_file_informations = []
        for c in classes:
            class_file_informations.append({
                "class" : c['name'],
                "indice" : c['indice'],
                "count_train" : c['meta']["count_train"],
                "count_val" : c['meta']["count_val"],
                "count_test" : c['meta']["count_test"],
            })

        np.random.shuffle(images_train)
        np.random.shuffle(images_val)
        np.random.shuffle(images_test)

        moment = f"{tools.get_format_date(pattern="%Y_%m_%d_%H_%M_%S")}"
        self.create_h5_dataset_img('train', images_train, to=moment)
        self.create_h5_dataset_img('val', images_val, to=moment)
        self.create_h5_dataset_img('test', images_test, to=moment)

        path_dir_moment = f"{config.__path_dir_datasets__}/{moment}/"
        self.create_class_file_informations(class_file_informations, path_dir_moment)

        datasets = self.read_h5_datasets(path_dir=path_dir_moment)

        return datasets, path_dir_moment

    def create_class_file_informations(self, datas, pathfile):
        """ Permet d'enregistrer les informations sur les classes """

        df = pd.DataFrame(datas)
        df.to_csv(f"{pathfile}classes.csv", index=False )
        self.__classes = df

    def load_class_file_informations(self, pathfile):
        """ Permet de charger le contenu d'un fichier d'information des classes du classifieur """

        if pathfile[-1] != "/":
            pathfile = pathfile + "/"

        df = pd.read_csv(f"{pathfile}classes.csv")
        self.__classes = df

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
            if file[-3:] == ".h5":
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

        self.load_class_file_informations(path_dir)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def show_images(self, x, classes=[], y=None, y_pred=None, indices="all", columns=10, fontsize=8, y_padding=1.35, limit=10, path_filename_to_save=None, spines_alpha=1, title=None):
        """ Permet d'afficher un listing d'images composant le set de données """
        if indices == "all":
            items = range(limit) if limit > -1 else range(len(x))
        else:
            items = indices if limit == -1 else indices[:limit]

        draw_labels = (y is not None)
        draw_pred = (y_pred is not None)

        rows = math.ceil(len(items)/columns)
        fig=plt.figure(figsize=(columns, rows*y_padding))
        n=1
        for i in items:
            axs=fig.add_subplot(rows, columns, n)
            n+=1

            img=axs.imshow(x[i])

            label = classes[y[i]] if y[i] < len(classes) else y[i]
            label_pred = classes[y_pred[i]] if y_pred is not None and y_pred[i] < len(classes) else y_pred[i] if y_pred is not None else ""

            if isinstance(label, bytes):
                label = str(label, encoding='utf-8')

            if isinstance(label_pred, bytes):
                label_pred = str(label_pred, encoding='utf-8')

            if draw_labels and not draw_pred:
                axs.set_xlabel( label,fontsize=fontsize)
            if draw_labels and draw_pred:
                if y[i] != y_pred[i]:
                    axs.set_xlabel(f'{ label_pred}\n({label})\n(i={i})',fontsize=fontsize)
                    axs.xaxis.label.set_color('red')
                else:
                    axs.set_xlabel(label,fontsize=fontsize)

            axs.set_yticks([])
            axs.set_xticks([])

            # gestion de la bordure d'image
            axs.spines['right'].set_visible(True)
            axs.spines['left'].set_visible(True)
            axs.spines['top'].set_visible(True)
            axs.spines['bottom'].set_visible(True)
            axs.spines['right'].set_alpha(spines_alpha)
            axs.spines['left'].set_alpha(spines_alpha)
            axs.spines['top'].set_alpha(spines_alpha)
            axs.spines['bottom'].set_alpha(spines_alpha)

        if title is not None:
            fig.suptitle(title, fontsize=14)

        if path_filename_to_save is not None:
            plt.savefig(path_filename_to_save)
        else:
            plt.show()





    def show_metrics(self, name, model=None, metrics={"Accuracy":['accuracy','val_accuracy'], 'Loss':['loss', 'val_loss']}, columns=2):
        """ Permet de visualiser les metrics de l'apprentissage du modèle """

        plt.figure(figsize=(12, 8))
        count_cols = len(metrics)
        count_rows = math.ceil(count_cols/columns)
        n = 1
        history = None

        # Historique depuis une instance de modèle entraîné
        if model is not None:
            history = model.history

        # Chargement depuis une fichier CSV
        if name is not None and model is None:
            history = self.load_history(name)

        if history is None:
            print("Historique non trouvé.")
            return False

        # for i, col in enumerate(columns):
        for title,curves in metrics.items():
            plt.subplot(count_rows, count_cols, n)
            plt.title(title)
            plt.ylabel(title)
            plt.xlabel('Epoch')
            for c in curves:
                plt.plot(history[c])

            plt.legend(curves, loc='upper left')
            n+=1

        plt.tight_layout()
        plt.show()

    def confusion_matrix(self, y_true, y_pred, labels):
        """ Permet d'afficher la matrice de confusion """

        title = f"Matrice de confusion"
        labels = [ str(y, encoding='utf-8') if isinstance(y,bytes) else y for y in labels ]

        y_true_converted = [ labels[y] for y in y_true ]
        y_pred_converted = [ labels[y] for y in y_pred ]

        cm = confusion_matrix( y_true_converted, y_pred_converted, normalize=None, labels=labels)

        # la somme des valeurs de la diagonale divisée par la somme totale de la matrice
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(15))
        fig = plt.figure(figsize=(15,15))
        axs = fig.add_subplot(1, 1, 1)
        axs.set_title(title, fontsize=40, pad=35)

        disp = ConfusionMatrixDisplay.from_predictions(y_true_converted, y_pred_converted, normalize='true', ax=axs)

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.tick_params(axis='x', rotation=45)
        plt.show()

        metrics = {
            'accuracy' : accuracy_score(y_true, y_pred),
            'precision' : precision_score(y_true, y_pred, average=None),
            'recall' : recall_score(y_true, y_pred,average=None),
            'f1' : f1_score(y_true, y_pred, average=None)
        }

        df = pd.DataFrame(metrics, index=labels)

        return df

    def show_predictions_errors(self, x, y_true, y_pred, classes=[] ):
        """ Permet de retourner les prédictions en erreurs """

        errors=[ i for i in range(len(x)) if y_pred[i]!=y_true[i] ]
        self.show_images(x, y=y_true, y_pred=y_pred, indices=errors, classes=classes)
        n_errors = len(errors)
        n_items = len(y_true)
        ratio = n_errors / n_items
        print(f"Nombre total d'erreur : {n_errors} / {n_items} ( {round( ratio ,4) * 100 } % ) ")

        return errors

    def show_layer_activation(self, model, img, name="", limit_layers="all", count_columns = 16, show_top=False, top = 3, verbose=1, labels=[]):
        """ Permet de visualiser les couches d'activations """

        fig = plt.figure(figsize=(2,2))
        axs = fig.add_subplot(1, 1, 1)
        # axs.imshow(img[...,::-1])
        axs.imshow(img)
        axs.set_yticks([])
        axs.set_xticks([])
        axs.set_title(f'Image : {name}')
        plt.show()

        shape = tuple(np.concatenate(([1], list(img.shape)), axis=0))

        layer_outputs = [layer.output for layer in model.layers ]
        activation_model = keras.models.Model(inputs=model.input,outputs=layer_outputs)
        activations = activation_model.predict(img.reshape(shape), verbose=verbose)

        layer_names = []
        for layer in model.layers:
            layer_names.append(layer.name)

        for layer_name, layer_activation in zip(layer_names, activations):

            try:
                if limit_layers != "all":
                    if limit_layers not in layer_name:
                        continue

                n_features = layer_activation.shape[-1]
                size = layer_activation.shape[1]
                n_cols = n_features // count_columns
                display_grid = np.zeros((size * n_cols, count_columns * size))
                scale = 1. / size

                for col in range(n_cols):
                    for row in range(count_columns):
                        channel_image = layer_activation[0,:,:, col * count_columns + row]
                        channel_image -= channel_image.mean()
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[col * size : (col + 1) * size, # Displays the grid
                                    row * size : (row + 1) * size] = channel_image


                if n_cols > 0:
                    fig = plt.figure(figsize=(scale * display_grid.shape[1],
                                        scale * display_grid.shape[0]))
                    axs = fig.add_subplot(1, 1, 1)
                    axs.imshow(img)
                    axs.set_yticks([])
                    axs.set_xticks([])
                    axs.set_title(f"{layer_name}, {layer_activation.shape}")
                    axs.imshow(display_grid, aspect='auto', cmap='viridis')
                    plt.show()

            except Exception as error:
                print(f"Impossible de générer la couche : {layer_name}")
                print(f"Erreur : {error}")
                print("")


        scores = activations[-1][0]
        indices = np.argsort(scores)[::-1][:top]
        names = [ labels[i] for i in indices ]
        p = activations[-1][0][indices]

        df = pd.DataFrame({'classe': names, 'score': p}, columns=['classe', 'score'])

        plt.bar(x= names , height=p)
        plt.title(f'Top {top} Predictions:')

        for i in range(len(names)):
            plt.text(i, p[i] , round(p[i], 2), ha = 'center')

        return df

    def show_infos_predictions(self, model, X, y, labels, indexes=[], alpha=0.005, limit_layers="conv", show_top=True, cam_path=None, limit=10, verbose=0):
        """ Permet de visualiser les informations de prédiction d'une ou plusieurs images (couches d'activations, tops, grad-cam) """

        for i, indice in enumerate(indexes):

            if i > limit and limit > -1:
                break

            df = self.show_layer_activation(model, img=X[indice], name= labels[y[indice]], limit_layers=limit_layers, show_top=show_top, labels=labels, verbose=verbose )

            print(df)

            image = X[indice]
            img_array = np.expand_dims(image, axis=0)

            preds = model.predict(img_array)
            c = np.argmax(preds[0])

            layers = [ l.name for l in model.layers if limit_layers in l.name ]
            last_conv_layer_name = layers[-1]
            heatmap = self.make_gradcam_heatmap(img_array, model, last_conv_layer_name, c)

            fig = plt.figure(figsize=(2,2))
            axs = fig.add_subplot(1, 1, 1)
            axs.matshow(heatmap, aspect='auto', cmap='viridis')
            axs.set_yticks([])
            axs.set_xticks([])
            axs.set_title('Heatmap')
            plt.show()

            self.save_and_display_gradcam(img_array, heatmap, alpha=alpha, cam_path=cam_path)

    def save_and_display_gradcam(self, img_array, heatmap, cam_path=None, alpha=0.5):
        # Load the original image
        # img = img_array[...,::-1].reshape(128,128,3)
        img = img_array.reshape(128,128,3)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet"]

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img_array = jet_heatmap * alpha + img
        superimposed_img = keras.utils.array_to_img(superimposed_img_array)

        # Save the superimposed image
        if cam_path is not None:
            superimposed_img.save(cam_path)
            # Display Grad CAM
            display(Image(cam_path))

        else:
            fig = plt.figure(figsize=(2,2))
            axs = fig.add_subplot(1, 1, 1)
            axs.imshow(superimposed_img)
            axs.set_yticks([])
            axs.set_xticks([])
            axs.set_title('Grad-CAM')
            plt.show()

    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        """ Permet de retourner une heatmap de la couche d'activation passée en paramètre """

        # Tout d'abord, nous créons un modèle qui mappe l'image d'entrée aux activations de la
        # dernière couche de conversion ainsi qu'aux prédictions de sortie.
        grad_model = keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Ensuite, nous calculons le gradient de la classe supérieure prédite pour notre image
        # d'entrée par rapport aux activations de la dernière couche de conversion.
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # Il s'agit du gradient du neurone de sortie (supérieur prédit ou choisi) par rapport à
        # la carte des caractéristiques de sortie de la dernière couche de conversion.
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Il s'agit d'un vecteur où chaque entrée correspond à l'intensité moyenne du gradient
        # sur un canal de carte de caractéristiques spécifique.
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Nous multiplions chaque canal du tableau de cartes de fonctionnalités par « l'importance de ce canal »
        # par rapport à la classe la plus prédite, puis additionnons tous les canaux pour obtenir l'activation de la classe Heatmap.
        # le symbole arobas "@" correspond à une multiplication de matrice
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # À des fins de visualisation, on normalise la carte thermique
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def load_history(self, name):
        """ Permet de charger l'historique d'un entraînement """
        return pd.read_csv(f'./runtime/{name}.history.log', sep=',', engine='python')