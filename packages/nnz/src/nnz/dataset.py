import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from IPython.display import display, Markdown
import math


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

    def showHeatmapNanValue(self, cbar=True):
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

    def showHistoVariable(self, columns=[], excludes=[], discrete=False, show_table=False):
        ''' Permet d'afficher un graphique de la distribution numérique des variables '''
        for col in columns:
            if col not in excludes and pd.api.types.is_numeric_dtype(self.df[col]):
                plt.figure(figsize=(15,8))
                sns.histplot(data=self.df, x=col, discrete=discrete, kde=True)
                plt.title(f"Distribution de la variable {col}")
                plt.show()

                if show_table == True:
                    values = self.df[col]
                    q = [np.percentile(values, p) for p in [25,50,75]]
                    n = len(values)

                    lorenz = self.getLorenz(values)
                    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
                    S = 0.5 - AUC
                    gini = 2*S

                    c = {
                        'taille' : n,
                        # Mesure de forme
                        'quartiles_25' : q[0],
                        'quartiles_50' : q[1],
                        'quartiles_75' : q[2],
                        'mean' : values.mean(),
                        'mean_square' : values.mean()**2,
                        'median' : values.median(),
                        'median_square' : values.median()**2,
                        'max' : values.max(),
                        'min' : values.min(),
                        'mode' : values.mode().loc[0],
                        'skew' : values.skew(),
                        'kurtosis' : values.kurtosis(),
                        # Mesure de dispersion
                        'variance' : values.var(),
                        'variance_corrige' : values.var(ddof=0),
                        'ecart-type' : values.std(),
                        'coef_variation' : values.std() / values.mean(),
                        'AUC' : AUC,
                        'S' : S,
                        'gini' : gini
                    }

                    print(f"Information de la variable {col}")
                    print(pd.DataFrame([c]))

    def showGridHistoVariable(self, columns=[], excludes=[]):
        ''' Permet d'afficher un ensemble de graphique de la distribution numérique des variables '''
        plt.figure(figsize=(12, 8))

        count_cols = 5
        count_rows = math.ceil(len(columns) / count_cols)

        for i, col in enumerate(columns):

            if col not in excludes and pd.api.types.is_numeric_dtype(self.df[col]):
                plt.subplot(count_rows, count_cols, i+1)
                sns.histplot(data=self.df, x=col, kde=True)

        plt.tight_layout()
        plt.show()

    def showCountValuesByVariable(self, ascending=True, by="values"):
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

    def showDistributionVariableQualitative(self, columns=[], excludes=[], show_table=False):
        """ Permet de visualiser la distribution empirique d'une variable via un graphe pie et bar"""

        def addvalues(ax, x,y):
            """ Ajout de la valeur en haut de la barre du graphique bar """
            for i in range(len(x)):
                ax.text(i, y[i]//2, y[i], ha = 'center')

        for col in columns:
            if col not in excludes:

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))

                t = self.df[col].value_counts(normalize=False).reset_index()

                axes[0].pie(t['count'], labels=t[col], autopct='%1.1f%%')

                axes[1].bar(height=t['count'], x=t[col], label=t[col])
                addvalues(axes[1], t[col], t['count'])

                fig.suptitle(f'Distribution empirique de la variable {col}')

                plt.show()

                if show_table == True:
                    tab = self.getDistributionVariable(col)
                    print(tab)

    def showGridDistributionVariableQualitative(self, columns=[], excludes=[], count_cols=3):
        """ Permet de visualiser la distribution empirique d'une variable via un graphe pie et bar"""

        plt.figure(figsize=(12, 8))
        count_rows = math.ceil(len(columns) / count_cols)
        c = 1
        for i, col in enumerate(columns):
            if col not in excludes :
                plt.subplot(count_rows, count_cols, c)
                t = self.df[col].value_counts(normalize=False).reset_index()
                plt.bar(height=t['count'], x=t[col], label=t[col])
                plt.title(f'Distribution de la variable {col}')
                c+=1

        plt.tight_layout()
        plt.show()


    def showDistributionVariableQualitativeHorizontal(self, columns=[], excludes=[], show_table=False, figsize=(15,15), limit_table=None):
        """ Permet de visualiser la distribution empirique d'une variable via un graphe bar horizontal """

        def addvalues(ax, x,y):
            """ Ajout de la valeur en haut de la barre du graphique bar """
            for i, v in enumerate(y):
                ax.text(x[i]+0.05, i, x[i], verticalalignment='center')

        for col in columns:
            if col not in excludes:
                fig, axes = plt.subplots(figsize=figsize)

                t = self.df[col].value_counts(normalize=False).reset_index()

                axes.barh(t[col], t['count'], align='center')
                addvalues(axes, t['count'], t[col])

                axes.invert_yaxis()  # labels read top-to-bottom
                axes.set_xlabel(f'Nombre')
                axes.set_title(f'Distribution empirique de la variable {col}')

                plt.show()

                if show_table == True:
                    tab = self.getDistributionVariable(col)
                    pd.set_option('display.max_rows', limit_table)
                    print(tab)
                    pd.set_option('display.max_rows', None)


    def showAllModalities(self, columns=[]):
        """ Permet de lister toutes les modalités de chaque variable qualitatives """

        for col in columns:
            print(f"Variable {col} : {self.df[col].unique()}")

    def getLorenz(self, values):
        """ Permet de retourner la courbe de Lorenz de la distribution d'une variable """
        lorenz = np.cumsum(np.sort(values)) / values.sum()
        lorenz = np.append([0],lorenz)
        return lorenz

    def showBoxPlotNumerical(self, columns=[], target="", excludes=[], count_box=4, vert=True):
        """ Permet d'afficher des boxplots par tranche de valeurs de la variable cible """
        for i, col in enumerate(columns):

            if col not in excludes:

                classes = []
                count_box = count_box + 1
                field_x = target
                taille = math.ceil(self.df[field_x].max() / count_box)
                field_y = col

                tranches = np.arange(0, max(self.df[field_x]), taille )
                indices = np.digitize(self.df[field_x], tranches)

                for i, tranche in enumerate(tranches):
                    # Ici on récupère les montants des dépenses pour notre tranche en fonction de la liste "indices"
                    montants = self.df.loc[ indices == i, field_y ]
                    if len(montants) > 0:
                        c = {
                            'valeurs' : montants,
                            'centre_classe' : tranche, #(tranches - (taille/2)) # On annule notre décalage sur l'axe x
                            'taille' : len(montants),
                            'quartiles' : [np.percentile(montants, p) for p in [25,50,75]]
                        }
                        classes.append(c)
                plt.figure(figsize=(10,7))

                valeurs = [c["valeurs"] for c in classes]
                positions = [c["centre_classe"] for c in classes]
                labels = []

                for i, c in enumerate(classes):
                    labels.append("{}\n(n={})".format(c['centre_classe'], c["taille"]))

                # affichage des boxplots
                # showfliers= False -> n'affiche pas les outliers
                fig, axe = plt.subplots(figsize=(15,8))
                axe.boxplot(valeurs,
                            positions= positions,
                            showfliers= True,
                            widths= taille*0.7,
                            labels=labels,
                            vert=vert)
                axe.set_xlabel(f'{field_x}')
                axe.set_ylabel(f'Observations de {field_y}')

                axe.set_title(f'Répartition par tranches de la variable {field_y} vs. {field_x}')
                plt.show()

    def showGridBoxplot(self, columns=[], excludes=[], y="", count_cols=3, vert=False):
        """ Permet d'afficher une grille de boîte à moustache """
        plt.figure(figsize=(15, 15))
        count_rows = math.ceil(len(columns) / count_cols)
        c = 1
        for i, col in enumerate(columns):
            if col not in excludes :
                plt.subplot(count_rows, count_cols, c)
                sns.boxplot(data=self.df, x=col, y=y)
                plt.title(f'{col} vs. {y}')
                c+=1

        plt.tight_layout()
        plt.show()

    def showBoxplotCategoricalNumerical(self,columns=[], excludes=[], target=""):
        """ Permet d'afficher un boxplot entre une variable qualitative et quantitative """
        for i, col in enumerate(columns):
            if col not in excludes:
                t = self.df[col].value_counts(normalize=False).reset_index()
                groupes = []
                labels = []
                for m in t.values:
                    groupes.append(self.df[self.df[col]==m[0]][target])
                    labels.append("{} (n={})".format(m[0], m[1]))

                # Propriétés graphiques (pas très importantes)
                medianprops = {'color':"black"}
                meanprops = {'markeredgecolor':'black',
                            'markerfacecolor':'firebrick'}

                figsize=(10, math.ceil(len(labels) / 2) )

                fig, axe = plt.subplots(figsize=figsize)
                axe.boxplot(groupes, labels=labels, showfliers=True, medianprops=medianprops,
                            vert=False, patch_artist=True, meanprops=meanprops)
                axe.yaxis.grid(True)
                axe.xaxis.grid(True)
                axe.set_xlabel(f'{target}')
                axe.set_ylabel(f'{col}')
                plt.show()