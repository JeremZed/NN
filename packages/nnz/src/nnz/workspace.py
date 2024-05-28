import platform
import pkg_resources
import nnz.tools as tools
from nnz.dataset import Dataset

import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import math
import pandas as pd
import pickle
import copy
import time
import cv2
import h5py


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error, explained_variance_score, root_mean_squared_error
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import learning_curve

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

    def saveModel(self, model, path, info=""):
        ''' Permet de sauvegarder un model '''
        m = copy.deepcopy(model)

        with open(path, 'wb') as f:
            pickle.dump(m, f)

    def loadModel(self, path):
        ''' Permet de charger un modèle depuis un fichier '''

        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

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

        self.saveModel(best_model, path_model)

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

    def split_folders(self, path_src, path_dst, size_train=0.6, size_val=0.15, size_test=0.25, limit=None, random_state=123, verbose=0, shuffle=True, resize=None  ):
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

        for c in classes_directories:
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

                dataset_train.append((img_arr, class_name))

            if verbose > 0:
                print(f"{len(train_files)} fichiers copiés dans {train_class_path}")

            for f in val_files:
                shutil.copy(f, val_class_path )
                img_arr = cv2.imread(f)
                if resize is not None:
                    img_arr = cv2.resize(img_arr,resize)
                dataset_val.append((img_arr, class_name))

            if verbose > 0:
                print(f"{len(val_files)} fichiers copiés dans {val_class_path}")

            for f in test_files:
                shutil.copy(f, test_class_path )
                img_arr = cv2.imread(f)
                if resize is not None:
                    img_arr = cv2.resize(img_arr,resize)
                dataset_test.append((img_arr, class_name))

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
