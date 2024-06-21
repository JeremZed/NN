import nnz.tools as tools
import nnz.config as config
from nnz.project.project import Project
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error, explained_variance_score, root_mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import pickle
import copy

import warnings
warnings.filterwarnings('ignore')

class ProjectML(Project):
    def __init__(self, name=None):
        super().__init__(name=name)

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

class ProjectMLClassifier(ProjectML):

    def __init__(self, name=None):
        super().__init__(name=name)


class ProjectMLPrediction(ProjectML):

    def __init__(self, name=None, target=None, verbose=1):
        super().__init__(name=name)
        self.verbose = verbose
        self.target = target

    def get_sets(self):
        """ Permet de retourner les differents sets (train, val, test) """
        return self.current_dataset.split(self.target, verbose=self.verbose)

    def evaluate_regression(self, model, X_data, y_data, y_pred):
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

    def show_graph(self, graphs=[] , count_cols = 2):
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

    def get_best_model(self, preprocessing, model_params, X_train, y_train, verbose=0):
        """ Permet de lancer un gridSearchCV sur un ensemble de modèle passés en paramètres """

        scores = []
        instances = {}

        path_dir = f'{config.__path_dir_runtime__}/{self.name}/{tools.get_format_date(pattern="%Y_%m_%d_%H_%M_%S")}'

        # Création du répertoire d'exécution
        tools.create_directory(path_dir)

        for model_name, mp in model_params.items():

            pipeline = make_pipeline(*preprocessing, mp['model'])
            if verbose > 0:
                print(f"[*] - Model : {model_name}")
                print(pipeline)

            clf =  GridSearchCV(pipeline, mp['params'], cv=5, return_train_score=True, refit=True, verbose=verbose)
            clf.fit(X_train, y_train)

            results = clf.cv_results_
            path_result_csv = f"{path_dir}/results_grid_csv_{model_name}.csv"
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
        path_model = f'{path_dir}/best_model.model'

        if verbose > 0:
            print(f"[*] - Saving model to {path_model}")

        self.save(best_model, path_model)

        return path_model, best_model, df