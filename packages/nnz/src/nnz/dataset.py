import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from IPython.display import display, Markdown


class Dataset():

    def __init__(self, data, t="array", sep=","):
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

    def getCountDuplicatedRows(self):
        """ Permet de retourner le nombre de ligne identique """
        return self.df.duplicated().sum()

    def showDuplicatedRowByVariable(self, column=None):
        """ Permet de retourner les lignes ayant les mêmes valeurs en fonction des colonnes passées en paramètre """

        if column is not None:
            display(Markdown(f'<b>Colonne : {column}</b>'))
            print(self.df.loc[ self.df[column].duplicated(keep=False), : ])
            print("\n")

    def getUniqueValueByVariable(self):
        """ Permet de récupérer le nombre de valeur unique par variable"""
        return self.df.nunique()

    def desc(self):
        """ Retourne la description statistique du dataset """
        return self.df.describe(include="all")

    def heatmapNanValue(self, cbar=True):
        ''' Permet de produire une image de l'ensemble du dataset pour visualiser les valeurs manquantes '''
        plt.figure(figsize=(20,10))
        plt.title("Représentation des valeurs manquante.")
        sns.heatmap( self.df.isna(), cbar=cbar )
        plt.show()

    def getRatioMissingValues(self, show_heatmap=False):
        ''' Permet de retourner le ratio de valeur manquante pour chaque variable '''

        a = self.df.isna().sum() / self.df.shape[0]
        df = pd.DataFrame(a, columns=['ratio'])
        df['sum'] = self.df.isna().sum()

        if show_heatmap == True:
            self.heatmapNanValue()

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

    def getDistributionVariable(self, col ):
        """ Permet de retourner numérique la distribution d'une variable du dataset """

        effectifs = self.df[col].value_counts()
        modalites = effectifs.index
        tab = pd.DataFrame(modalites, columns=[col])
        tab['n'] = effectifs.values
        tab['f'] = tab['n'] / len(self.df)
        tab['F'] = tab['f'].cumsum()

        return tab

    def showDistributionVariableQualitative(self, col, show_table=False):
        """ Permet de visualiser la distribution empirique d'une variable via un graphe pie et bar"""

        def addlabels(ax, x,y):
            """ Ajout de la valeur en haut de la barre du graphique bar """
            for i in range(len(x)):
                ax.text(i, y[i]//2, y[i], ha = 'center')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))

        t = self.df[col].value_counts(normalize=False).reset_index()

        axes[0].pie(t['count'], labels=t[col], autopct='%1.1f%%')

        axes[1].bar(height=t['count'], x=t[col], label=t[col])
        addlabels(axes[1], t[col], t['count'])

        fig.suptitle(f'Distribution empirique de la variable {col}')

        plt.show()

        if show_table == True:
            tab = self.getDistributionVariable(col)
            print(tab)

    def showDistributionVariableQualitativeHorizontal(self, col, show_table=False, figsize=(15,15), limit_table=None):
        """ Permet de visualiser la distribution empirique d'une variable via un graphe bar horizontal """

        def addlabels(ax, x,y):
            """ Ajout de la valeur en haut de la barre du graphique bar """
            for i, v in enumerate(y):
                ax.text(x[i]+0.05, i, x[i], verticalalignment='center')

        fig, axes = plt.subplots(figsize=figsize)

        t = self.df[col].value_counts(normalize=False).reset_index()

        axes.barh(t[col], t['count'], align='center')
        addlabels(axes, t['count'], t[col])

        axes.invert_yaxis()  # labels read top-to-bottom
        axes.set_xlabel(f'Nombre')
        axes.set_title(f'Distribution empirique de la variable {col}')

        plt.show()

        if show_table == True:
            tab = self.getDistributionVariable(col)
            pd.set_option('display.max_rows', limit_table)
            print(tab)
            pd.set_option('display.max_rows', None)