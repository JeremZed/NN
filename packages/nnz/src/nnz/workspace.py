import platform
import pkg_resources
import nnz.tools as tools
import nnz.config as config
from nnz.dataset import Dataset

import numpy as np
import os
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import pandas as pd
import pickle
import copy
import time
import cv2
import h5py
from IPython.display import Image, display
from PIL import Image as im


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error, explained_variance_score, root_mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import learning_curve

os.environ['KERAS_BACKEND'] = config.__env_keras__
import keras
import tensorflow as tf

class Workspace():
    """
    Représente le projet en cours et son environnement
    """
    def __init__(self):

        self.__dir_runtime = tools.create_directory("./runtime")
        self.__dir_resources = tools.create_directory("./resources")
        self.__date_init = tools.get_current_date()
        self.__platform = platform.uname()
        self.__datasets = []
        self.__classes = pd.read_csv('./resources/classes.csv') if os.path.exists('./resources/classes.csv') else pd.DataFrame([])

    def get_classes(self):
        """ Permet de retourner la liste des classes """
        return self.__classes

    def get_date_init(self):
        """ Permet de retourner la date où le workspace a été initialisé """
        return self.__date_init

    def show_informations(self):
        """ Permet d'afficher l'ensemble des informations du worskpace """

        print(f'\n{tools.get_fill_string()}')
        print(f"- Date : {self.__date_init}")
        print(f"- Répertoire runtime : {self.__dir_runtime}")
        print(f"- Machine : {self.__platform}")
        print(f'{tools.get_fill_string()}\n')
        print(f'\n{tools.get_fill_string()}')
        print(f"Liste des modules python installés :")
        print(f'{tools.get_fill_string()}\n')

        installed_packages = pkg_resources.working_set
        for package in installed_packages:
            print(f"{package.key}=={package.version}")


    def add_dataset(self, name, dataset,  **kwargs):
        """ Permet d'ajouter un nouveau dataset au work """
        self.__datasets.append({ "name" : name, "dataset" : Dataset(dataset, **kwargs) })
        return self.get_dataset(name)

    def clear_datasets(self):
        """ Permet de vider la liste des datasets """
        self.__datasets = []

    def remove_dataset(self, name):
        """ Permet de supprimer un dataset de la liste """
        d = tools.get_item("name", name, self.__datasets)
        if d is not None:
            del self.__datasets[d[0]]

    def get_dataset(self, name="__all__"):
        """ Getter de l'attribut __datasets """
        if name == "__all__":
            return self.__datasets
        else:
            d = tools.get_item("name", name, self.__datasets)
            return d[1]['dataset'] if d is not None else None


    def evaluateRegression(self, model, X_data, y_data, y_pred):
        """ Permet d'afficher les métrics pour une régression"""

        mse = mean_squared_error(y_data, y_pred)
        r2_square = r2_score(y_data,y_pred)
        mae = mean_absolute_error(y_data, y_pred)
        sp = spearmanr(y_pred, y_data).correlation
        pe = pearsonr(y_pred, y_data).correlation
        ex = explained_variance_score(y_data, y_pred)
        score = model.score(X_data, y_data)
        rmse = root_mean_squared_error(y_data, y_pred)

        print(f"R2: {r2_square}")
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'Spearman: {sp}')
        print(f'Pearson: {pe}')
        print(f'Variance: {ex}')
        print(f'Score: {score}')

    def learning_curve(self, model, X, y):
        """ Permet d'afficher la courbe d'apprentissage du modèle """

        N, train_score, val_score = learning_curve(model, X, y, train_sizes=np.linspace(0.1,1,10) )

        plt.figure(figsize=(12,8))
        plt.plot(N, train_score.mean(axis=1), label="train score")
        plt.plot(N, val_score.mean(axis=1), label="validation score")
        plt.title(f'Learning curve avec le model {model}')
        plt.legend()
        plt.show()

    def showGraphPrediction(self, graphs=[] , count_cols = 2):
        """ Permet d'afficher un graphique représentant le positionnement des prédictions par rapport à la réalité """

        plt.figure(figsize=(12, 6))

        count_rows = math.ceil(len(graphs) / count_cols)

        idx = 1
        for graph in graphs:

            true_data, predict_data, color, title = graph

            plt.subplot(count_rows, count_cols, idx)
            plt.scatter(true_data, predict_data, c=color, label='Predicted')
            plt.plot([min(true_data), max(true_data)], [min(true_data), max(true_data)], '--k', lw=2)
            plt.title(title)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")

            idx+=1

        plt.tight_layout()
        plt.show()

    def save(self, object, path, info=""):
        ''' Permet de sauvegarder le contenu d'un objet '''
        m = copy.deepcopy(object)

        with open(path, 'wb') as f:
            pickle.dump(m, f)

    def load(self, path):
        ''' Permet de charger un objet depuis un fichier '''

        with open(path, 'rb') as f:
            object = pickle.load(f)

        return object

    def fitModel(self, model, X, y, X_val, y_val, name='model', batch_size=16, epochs=1, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1 ):
        """ Permet de lancer le fit du modèle avec les différents callback de sauvegarde du modèle """

        checkpoint_filepath = f'./runtime/ckpt/checkpoint.{name}.keras'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only)

        csv_logger = keras.callbacks.CSVLogger(f'./runtime/{name}.history.log')

        with open(f'./runtime/{name}.summary.json', "w") as json_file:
            json_file.write(model.to_json())

        return model.fit(  X, y,
                      batch_size      = batch_size,
                      epochs          = epochs,
                      verbose         = verbose,
                      validation_data = (X_val, y_val),
                      callbacks=[model_checkpoint_callback, csv_logger])

    def loadSummaryModel(self, name):
        """ Permet de retourner l'architecture d'un modèle """

        f = open(f'./runtime/{name}.summary.json', 'r')
        config = f.read()
        f.close()
        model = keras.models.model_from_json(config)

        return model.summary()

    def loadModel(self, name):
        """ Permet de charger un modèle """

        checkpoint_filepath = f'./runtime/ckpt/checkpoint.{name}.keras'
        return keras.models.load_model(checkpoint_filepath)

    def loadHistory(self, name):
        """ Permet de charger l'historique d'un entraînement """
        return pd.read_csv(f'./runtime/{name}.history.log', sep=',', engine='python')

    def getBestModel(self, preprocessing, model_params, X_train, y_train, verbose=0):
        """ Permet de lancer un gridSearchCV sur un ensemble de modèle passés en paramètres """

        scores = []
        instances = {}

        for model_name, mp in model_params.items():

            pipeline = make_pipeline(*preprocessing, mp['model'])
            if verbose > 0:
                print(f"[*] - Model : {model_name}")
                print(pipeline)

            clf =  GridSearchCV(pipeline, mp['params'], cv=5, return_train_score=True, refit=True, verbose=verbose)
            clf.fit(X_train, y_train)

            results = clf.cv_results_
            path_result_csv = f"./runtime/results_grid_csv_{model_name}.csv"
            _df = pd.DataFrame(results)
            _df.to_csv(path_result_csv)
            if verbose > 0:
                print(f"[*] - Saving grid result to {path_result_csv}")

            if "show_learning_curve" in mp and mp['show_learning_curve'] == True:
                self.learning_curve(clf.best_estimator_, X_train, y_train)
                # self.showGraphParamGrid(clf, model_name, _df) # BETA

            scores.append({
                'model': model_name,
                'best_score': clf.best_score_,
                'best_params': clf.best_params_
            })

            instances[model_name] = clf
            if verbose > 0:
                print("\n")

        if verbose > 0:
            print("[*] - Done.")

        df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
        df = df.sort_values(by=['best_score'], ascending=False)

        best_model_name = df.iloc[0]['model']
        best_model = instances[best_model_name]
        path_model = f'./runtime/best_model_{tools.get_format_date(pattern="%d_%m_%Y_%H_%M_%S")}.model'

        if verbose > 0:
            print(f"[*] - Saving model to {path_model}")

        self.save(best_model, path_model)

        return path_model, best_model, df


    def showGraphParamGrid(self, clf, model_name, _df):
        """ Permet de visualiser les courbes de scores pour chaque paramètre d'un grid search CV """

        print("DICT PARAMS : ", clf.param_grid)
        params = clf.param_grid

        fig, ax = plt.subplots(1,len(params),figsize=(20,5))
        fig.suptitle(f'Score per parameter of {model_name}')
        fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')

        for i, p in enumerate(params):
            # print("--> : ", p, params[p])
            name = p.split("__")

            x = np.array([ str(a) for a in params[p] ])
            axis_y_train = []
            axis_y_test = []
            y_train_e = []
            y_test_e = []

            for param_value in params[p]:
                # print(f"Param Value : {param_value}")
                values_train = _df[_df[f"param_{p}"] == param_value]['mean_train_score'].agg(['min', 'max', 'mean'])
                values_train = np.where(np.isnan(values_train), 0, np.array(values_train))
                axis_y_train.append(values_train[2])
                y_train_e.append(values_train[1] - values_train[0])

                values_test = _df[_df[f"param_{p}"] == param_value]['mean_test_score'].agg(['min', 'max', 'mean'])
                values_test = np.where(np.isnan(values_test), 0, np.array(values_test))
                axis_y_test.append(values_test[2])
                y_test_e.append(values_test[1] - values_test[0])

            # print("----->" , x, y,y_e)

            if len(params) == 1:
                ax.errorbar(
                    x=x,
                    y=axis_y_train,
                    yerr=y_train_e,
                    label='train_score'
                )
                ax.errorbar(
                    x=x,
                    y=axis_y_test,
                    yerr=y_test_e,
                    label='test_score'
                )
                ax.set_xlabel(name[1].upper())
            else:
                ax[i].errorbar(
                    x=x,
                    y=axis_y_train,
                    yerr=y_train_e,
                )
                ax[i].errorbar(
                    x=x,
                    y=axis_y_test,
                    yerr=y_test_e,
                )
                ax[i].set_xlabel(name[1].upper())

        plt.show()

    def splitFolders(self, path_src, path_dst, size_train=0.6, size_val=0.15, size_test=0.25, limit=None, random_state=123, verbose=0, shuffle=True, resize=None  ):
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
        classes = []

        dataset_train = []
        dataset_val = []
        dataset_test = []

        for i, c in enumerate(classes_directories):
            class_name = c.split("/")[-1]

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

            files = tools.list_files(c)

            if limit is not None and limit < len(files):
                files = files[:limit]

            if shuffle == True:
                np.random.shuffle(files)

            total_size = len(files)

            train_files, val_files, test_files = np.split(
                files,
                [
                    int(total_size*size_train),
                    int(total_size*(size_train + size_val))
                ])

            classes.append({
                "name" : class_name,
                "train_count" : len(train_files),
                "val_count" : len(val_files),
                "test_count" : len(test_files),
            })

            if verbose > 0:
                print(class_name, len(files), "fichiers")
                print("size_train : " , len(train_files), f"({size_train})")
                print("size_val : ", len(val_files), f"({size_val})")
                print("size_test : ", len(test_files), f"({size_test})")
                print("size_total : ", (len(train_files) + len(val_files) + len(test_files)), f"({(size_train + size_val + size_test)})")

            for f in train_files:
                shutil.copy(f, train_class_path )
                img_arr = cv2.imread(f)
                if resize is not None:
                    img_arr = cv2.resize(img_arr,resize)
                    img_arr = img_arr[...,::-1]

                dataset_train.append((img_arr, i))

            if verbose > 0:
                print(f"{len(train_files)} fichiers copiés dans {train_class_path}")

            for f in val_files:
                shutil.copy(f, val_class_path )
                img_arr = cv2.imread(f)
                if resize is not None:
                    img_arr = cv2.resize(img_arr,resize)
                    img_arr = img_arr[...,::-1]
                dataset_val.append((img_arr, i))

            if verbose > 0:
                print(f"{len(val_files)} fichiers copiés dans {val_class_path}")

            for f in test_files:
                shutil.copy(f, test_class_path )
                img_arr = cv2.imread(f)
                if resize is not None:
                    img_arr = cv2.resize(img_arr,resize)
                    img_arr = img_arr[...,::-1]
                dataset_test.append((img_arr, i))

            if verbose > 0:
                print(f"{len(test_files)} fichiers copiés dans {test_class_path}")
                print("\n")

        if verbose > 0:
            print("[*] - Terminé.")

        # On sauvegarde dans un fichier la liste des classes
        df = pd.DataFrame(classes)
        df.to_csv("./resources/classes.csv", index=False )
        self.__classes = df

        np.random.shuffle(dataset_train)
        np.random.shuffle(dataset_val)
        np.random.shuffle(dataset_test)

        X_train = [ x[0] for x in dataset_train]
        X_val = [ x[0] for x in dataset_val]
        X_test = [ x[0] for x in dataset_test]

        y_train = [ x[1] for x in dataset_train]
        y_val = [ x[1] for x in dataset_val]
        y_test = [ x[1] for x in dataset_test]

        # On sauvegarde dans un fichier le dataset
        hf = h5py.File('./resources/dataset.h5', 'w')
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('X_val', data=X_val)
        hf.create_dataset('y_val', data=y_val)
        hf.create_dataset('X_test', data=X_test)
        hf.create_dataset('y_test', data=y_test)
        hf.close()

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

    def loadDatasetH5(self, path_file):
        """ Permet de charger le contenu d'un dataset contenu dans un fichier .h5 """

        if os.path.exists(path_file) == False:
            raise Exception(f"Fichier inconnu : {path_file}")

        return h5py.File(path_file, 'r')

    def loadDatasetSplited(self, path_file):
        """ Permet de récupérer le contenu splité (x_train, y_train etc...) """

        hf = self.loadDatasetH5(path_file=path_file)

        X_train = hf.get('X_train')
        y_train = hf.get('y_train')
        X_val = hf.get('X_val')
        y_val = hf.get('y_val')
        X_test = hf.get('X_test')
        y_test = hf.get('y_test')

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

    def loadGroupsDataset(self, path_file, groups=None):
        """ Permet de retourner l'ensemble des groupes qui composent un dataset au format h5 """

        hf = self.loadDatasetH5(path_file=path_file)
        hf_groups = hf.keys()

        items = []
        if groups is not None:
            for g in groups:
                if g in hf_groups:
                    items.append(hf.get(g))
        else:
            for g in hf_groups:
                items.append(hf.get(g))

        return items

    def showImages(self, x, classes=[], y=None, y_pred=None, indices="all", columns=10, fontsize=8, y_padding=1.35, limit=10, path_filename_to_save=None, spines_alpha=1, title=None):
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
            # On affiche l'image avec les couleurs d'origine en convertissant le BGR produit par cv2 lors de la création vers RGB
            # img=axs.imshow(cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB)) pas compatible avec la normalisation des datas entre 0 et 1
            # img=axs.imshow(x[i][...,::-1])
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

    def showMetrics(self, name, model=None, metrics={"Accuracy":['accuracy','val_accuracy'], 'Loss':['loss', 'val_loss']}, columns=2):
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
            history = self.loadHistory(name)

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

    def confusionMatrix(self, y_true, y_pred, labels):
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

    def showPredictionsErrors(self, x, y_true, y_pred, classes=[] ):
        """ Permet de retourner les prédictions en erreurs """

        errors=[ i for i in range(len(x)) if y_pred[i]!=y_true[i] ]
        self.showImages(x, y=y_true, y_pred=y_pred, indices=errors, classes=classes)
        n_errors = len(errors)
        n_items = len(y_true)
        ratio = n_errors / n_items
        print(f"Nombre total d'erreur : {n_errors} / {n_items} ( {round( ratio ,4) * 100 } % ) ")

        return errors

    def showLayerActivation(self, model, img, name="", limit_layers="all", count_columns = 16, show_top=False, top = 3, verbose=1, labels=[]):
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

    def showInfosPredictions(self, model, X, y, labels, indexes=[], alpha=0.005, limit_layers="conv", show_top=True, cam_path=None, limit=10, verbose=0):
        """ Permet de visualiser les informations de prédiction d'une ou plusieurs images (couches d'activations, tops, grad-cam) """

        for i, indice in enumerate(indexes):

            if i > limit and limit > -1:
                break

            df = self.showLayerActivation(model, img=X[indice], name= labels[y[indice]], limit_layers=limit_layers, show_top=show_top, labels=labels, verbose=verbose )

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