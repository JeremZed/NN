import platform
import pkg_resources
import nnz.tools as tools
from nnz.dataset import Dataset

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import pickle
import copy
import time


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