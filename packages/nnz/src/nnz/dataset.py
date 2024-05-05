import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Dataset():

    def __init__(self, data, t="array", sep=",") -> None:
        self.is_dataframe = False

        if isinstance(data, str) :
            if t == "csv":
                self.df = self.__loadDatasetCSV(data, sep=sep)
        else:
            if len(data.shape) < 2:
                if t == "array":
                    self.df = self.__create(data)
                else:
                    self.df = self.__create([])
            else:
                self.df = np.array(data)

    def __create(self, data):
        """ Permet de créer un dataset """
        self.is_dataframe = True
        return pd.DataFrame(data)

    def __loadDatasetCSV(self, path, sep=","):
        ''' Permet de charger le dataset en fonction du chemin du fichier CSV passé en paramètre via pandas'''
        self.is_dataframe = True
        return pd.read_csv(path, sep=sep)

    def getCountRowColumns(self):
        ''' Permet de retourner le nombre de ligne et de colonne du dataset '''
        s = self.df.shape
        return { 'rows' : s[0], 'columns' : s[1] }

    def getTypesVariables(self, type="all"):
        ''' Permet de retourner les types des variables du dataset '''
        if type == "count":
            return self.df.dtypes.value_counts()
        else:
            return self.df.dtypes

    def heatmapNanValue(self, cbar=True):
        ''' Permet de produire une image de l'ensemble du dataset pour visualiser les valeurs manquantes '''
        plt.figure(figsize=(20,10))
        plt.title("Représentation des valeurs manquante.")
        sns.heatmap( self.df.isna(), cbar=cbar )
        plt.show()

    def getRatioMissingValues(self):
        ''' Permet de retourner le ratio de valeur manquante pour chaque variable '''

        a = self.df.isna().sum() / self.df.shape[0]
        df = pd.DataFrame(a, columns=['ratio'])
        df['sum'] = self.df.isna().sum()

        return df.sort_values('ratio', ascending=False)

    def getHistoVariable(self, column=None):
        ''' Permet d'afficher un graphique de répartition des données en fonction de leur type '''
        for col in self.df.columns:
            show = False

            if column is not None and col == column:
                show = True

            if column is None:
                show = True

            if show == True:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    plt.figure(figsize=(15,8))
                    center = False
                    if self.df[col].dtype in ['uint8']:
                        center = True

                    sns.histplot(data=self.df, x=col, discrete=center)
                    plt.show()

    def getCountValuesByVariable(self, ascending=True, by="values"):
        """ Permet d'afficher le nombre d'occurence ayant les mêmes valeurs pour chaque variable """
        if isinstance(self.df, pd.DataFrame) :
            for col in self.df.columns:
                print(self.df[col].value_counts().sort_values(ascending=ascending))
        else:
            print("Fonctionne uniquement avec un DataFrame")

    def normalize(self):
        """ Permet de normaliser le dataset """
        if self.is_dataframe == True:
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(self.df),columns = self.df.columns)
            return df_scaled
        else:
            xmax = self.df.max()
            return self.df / xmax
