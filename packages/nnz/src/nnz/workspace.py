import platform
import pkg_resources
import nnz.tools as tools
from nnz.dataset import Dataset

import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error, explained_variance_score, root_mean_squared_error
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import learning_curve

MODEL_LinearRegression = "LinearRegression"
MODEL_RandomForestRegressor = "RandomForestRegressor"
MODEL_GradientBoostingRegressor = "GradientBoostingRegressor"
MODEL_SVR = "SVR"
MODEL_XGBRegressor = "XGBRegressor"

class _LinearRegression(LinearRegression):
    pass

class _RandomForestRegressor(RandomForestRegressor):
    pass

class _GradientBoostingRegressor(GradientBoostingRegressor):
    pass

class _SVR(SVR):
    pass

class _XGBRegressor(XGBRegressor):
    pass

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

    def model(self, name, **kwargs):
        """ Factory de modèle """
        model = None
        if name == MODEL_LinearRegression:
            model = _LinearRegression(**kwargs)
        if name == MODEL_RandomForestRegressor:
            model = _RandomForestRegressor(**kwargs)
        if name == MODEL_GradientBoostingRegressor:
            model = _GradientBoostingRegressor(**kwargs)
        if name == MODEL_SVR:
            model = _SVR(**kwargs)
        if name == MODEL_XGBRegressor:
            model = _XGBRegressor(**kwargs)

        if model is None:
            print(f'[*] - Modèle non implémenté.')

        return model

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

            true_data, predict_data, color = graph

            plt.subplot(count_rows, count_cols, idx)
            plt.scatter(true_data, predict_data, c=color, label='Predicted')
            plt.plot([min(true_data), max(true_data)], [min(true_data), max(true_data)], '--k', lw=2)
            plt.title("Training Results")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")

            idx+=1

        plt.tight_layout()
        plt.show()
