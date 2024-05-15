import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# from scipy.stats import spearmanr, zscore
import numpy as np
from IPython.display import display, Markdown
import math
import scipy.stats as st


class Dataset():

    def __init__(self, data, t="array", sep=","):
        self.is_dataframe = False
        self.rollback_scaler = None

        if isinstance(data, str) :
            if t == "csv":
                self.df = self.__loadDatasetCSV(data, sep=sep)
        else:
            if len(data.shape) <= 2:
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

    def calculateOutliersZscore(self, columns=[], threshold_z = 3):
        """ Permet de retourner un df représentant les outliers dans les colonnes passées en paramètres """
        df = st.zscore(self.df[columns])
        d = []
        for c in df.columns:
            count = len(df[ np.abs(df[c]) > threshold_z ]  )
            max = df[c].max()
            min = df[c].min()
            ratio = round(count / self.df.shape[0], 2)

            d.append( { 'variable' : c, 'count' : count, 'max' : max, 'min' : min, 'ratio' : ratio } )
        return pd.DataFrame(d).set_index('variable').sort_values('count', ascending=False)

    def showBoxplot(self, columns=[]):
        """ Permet d'afficher une liste de boîte à moustache pour chaque colonne """
        for col in self.df[columns].columns:
            sns.boxplot(self.df[col], showfliers= True)
            plt.show()

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
                        # Mesure de tendance centrale
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
                        # Mesure de forme
                        'skew' : values.skew(),
                        'kurtosis' : values.kurtosis(),
                        # Mesure de dispersion
                        'variance' : values.var(),
                        'variance_corrige' : values.var(ddof=0),
                        'ecart-type' : values.std(),
                        'coef_variation' : values.std() / values.mean(),
                        # Mesure de concentration
                        'lorenz' : lorenz.sum(),
                        'AUC' : AUC,
                        'Surface' : S,
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

    def showBoxPlotNumerical(self, columns=[], target="", excludes=[], count_box=4, show_table=False):
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

                    items = self.df.loc[ indices == i, field_y ]
                    if len(items) > 0:
                        c = {
                            'valeurs' : items,
                            'centre_classe' : tranche,
                            'taille' : len(items),
                            'quartiles' : [np.percentile(items, p) for p in [25,50,75]]
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
                            labels=labels)
                axe.set_xlabel(f'{field_x}')
                axe.set_ylabel(f'Observations de {field_y}')

                axe.set_title(f'Répartition par tranches de la variable {field_y} vs. {field_x}')
                plt.show()

                if show_table == True:
                    pearson = st.pearsonr(self.df[field_x],self.df[field_y])
                    covariance = np.corrcoef(self.df[field_x],self.df[field_y])
                    a, b = np.polyfit(self.df[field_x],self.df[field_y], 1)
                    d = {
                        "pearson_statistic" : pearson[0],
                        "pearson_pvalue" : pearson[1],
                        "correlation" : covariance[0][1],
                        "a" : a,
                        "b" : b
                    }

                    fig, (axe1) = plt.subplots(1,1, figsize=(15,8))

                    x_n = int(self.df[field_x].max()) + 1

                    axe1.plot(self.df[field_x],self.df[field_y], 'o')
                    axe1.plot(np.arange( x_n ), [a*x+b for x in np.arange(x_n)])
                    axe1.set_xlabel(f'{field_x}')
                    axe1.set_ylabel(f'{field_y}')
                    axe1.set_title(f'Regression linaire de {field_y} vs {field_x}')


                    print(pd.DataFrame([d]))
                    print("\n")
                    print("-"*100)
                    print("\n")

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

    def getCorrelationPearson(self, columns=[], excludes=[], target="", fillna=False, with_grid=False, count_cols=3):
        ''' Permet de retourner une vision de corrélation Pearson entre les variables et la target dans le cas variable continue / continue '''
        res = []
        data = self.df.copy()

        if fillna == True:
            data = data.fillna(0)

        for col in columns:
            if col not in excludes:
                p = st.pearsonr(data[col], data[target] )
                res.append( {"name" : col, "statistic" : p[0], "pvalue" : p[1] } )

        df = pd.DataFrame(res).sort_values('statistic')

        if with_grid == True:
            plt.figure(figsize=(15, 15))
            count_rows = math.ceil(len(columns) / count_cols)
            c = 1
            for row in df.values:

                col = row[0]
                if col not in excludes :

                    plt.subplot(count_rows, count_cols, c)

                    x_n = int(self.df[target].max()) + 1
                    a, b = np.polyfit(self.df[target],self.df[col], 1)

                    plt.plot(self.df[target],self.df[col], 'o')
                    plt.plot(np.arange( x_n ), [a*x+b for x in np.arange(x_n)])
                    plt.xlabel(f'{target}')
                    plt.ylabel(f'{col}')
                    plt.title(f'Regression linaire de {col} vs {target}')

                    c+=1

            plt.tight_layout()
            plt.show()

        return df

    def showCrossTab(self, variables, target):
        ''' Permet de visualiser la crosstab de pandas en fonctions des variables qualitatives '''
        for col in variables:
            print( pd.crosstab(self.df[target], self.df[col]) )


    def getHeatCrossTab(self, variables, target):
        ''' Permet de visualiser une heatmap des crosstab des variables qualitatives '''
        for col in variables:
            plt.figure()
            sns.heatmap( pd.crosstab(self.df[target], self.df[col]), annot=True, fmt="d" )
            plt.pause(0.001)

        plt.show(block=True)

    def getCorrelationCategoricalNumerical(self,columns=[], excludes=[], target=""):
        """ Permet de retourner un tableau de rapport de corrélation entre les variables qualitative et une variable quantitative """
        def eta_squared(x,y):
            moyenne_y = y.mean()
            classes = []
            for classe in x.unique():
                yi_classe = y[x==classe]
                classes.append({'ni': len(yi_classe),
                                'moyenne_classe': yi_classe.mean()})
            SCT = sum([(yj-moyenne_y)**2 for yj in y])
            SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
            rapport = SCE/SCT

            return (SCT, SCE, rapport)

        res=[]
        for i, col in enumerate(columns):
            if col not in excludes:
                r = eta_squared(self.df[col],self.df[target])
                res.append({ 'name' : col, 'SCT' : r[0], 'SCE' : r[1], 'rapport de corrélation' : r[2] })

        return pd.DataFrame(res)

    def getHeatCorrelationNumerical(self, columns, cluster=False):
        ''' Permet d'afficher une heatmap des corrélations entre variables continue '''
        plt.figure(figsize=(15,8))
        sns.heatmap(self.df[ columns ].corr(), annot=True)
        if cluster == True:
            sns.clustermap(self.df[ columns ].corr(), annot=True)

    def getHeatCorrelationCategorical(self, columns=[], excludes=[], count_cols=2, cluster=False):

        for i, _col in enumerate(columns):
            if _col not in excludes :

                target = _col

                plt.figure(figsize=(15, 240 ))
                count_rows = math.ceil(len(columns) / count_cols) * len(columns)
                idx = 1

                for col in columns :
                    if _col != col and col not in excludes:
                        # print(count_rows, count_cols, idx)
                        axe = plt.subplot(count_rows, count_cols, idx)

                        cont = self.df[[target,col]].pivot_table(index=target,columns=col,aggfunc=len,margins=True,margins_name="Total")
                        tx = cont.loc[:,["Total"]]
                        ty = cont.loc[["Total"],:]
                        n = len(self.df)
                        indep = tx.dot(ty) / n

                        c = cont.fillna(0) # On remplace les valeurs nulles par 0
                        measure = (c-indep)**2/indep
                        xi_n = measure.sum().sum()
                        table = measure/xi_n
                        sns.heatmap(table.iloc[:-1,:-1],annot=True, ax=axe)

                        idx += 1
                plt.tight_layout()
                plt.show()
                print("\n")
                print("-"*100)
                print("\n")

    def split(self, target, size_train=0.6, size_val=0.15, size_test=0.25, random_state=123, stratify=None, verbose=0):
        """ Permet de spliter un df en 3 partie train, validation, test  """
        if size_train + size_val + size_test != 1.0:
            raise ValueError( f'Le cumul de size_train:{size_train}, size_val:{size_val}, size_test:{size_test} n\'est pas égal à 1.0')

        if target not in self.df.columns:
            raise ValueError(f'La colonne : {target} n\'est pas présente dans le dataframe')

        X = self.df.drop(target, axis=1)
        y = self.df[target]

        y_stratify = None
        if stratify is not None:
            y_stratify = y

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y_stratify, test_size=(1.0 - size_train), random_state=random_state)

        size_rest = size_test / (size_val + size_test)

        y_stratify_temp = None
        if stratify is not None:
            y_stratify_temp = y_temp

        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_stratify_temp,test_size=size_rest,random_state=random_state)

        assert len(self.df) == len(X_train) + len(X_val) + len(X_test)

        if verbose > 0:
            print(f"Ratio : size_train={size_train}, size_val={size_val}, size_test={size_test}")
            print(f"Shape Trainset => X : {X_train.shape} y : {y_train.shape}")
            print(f"Shape Valset => X : {X_val.shape} y : {y_val.shape}")
            print(f"Shape Testset => X : {X_test.shape} y : {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test


    def getCategoricalVariableEncoded(self, columns=[], excludes=[], encoder="one-hot"):
        """ Permet d'encoder des variables qualitatives """

        columns_to_hot = [ item for item in columns if item not in excludes ]
        output = None

        if encoder == "one-hot":
            cat_encoder = OneHotEncoder()
            encoded = cat_encoder.fit_transform(self.df[ columns_to_hot ])

            output = pd.DataFrame(encoded.toarray(), columns=cat_encoder.get_feature_names_out(), index=self.df[ columns_to_hot ].index)


        if output is not None:
            df = pd.concat([self.df, output], axis=1)
            df = df.drop(columns_to_hot,axis=1)

            return df

        return output

    def scaler(self, mode="min-max", excludes=[], verbose=0):
        """ Permet de centrer/réduire ou de normaliser les datas sans faire de copie """
        self.rollback_scaler = self.df.copy()
        cols = [ item for item in self.df.columns if item not in excludes ]

        if mode == "custom-normalizer":
            self.df_exclued = self.df[excludes].copy()
            self.df = (self.df[cols] ) / self.df[cols].std()
            self.df[excludes] = self.df_exclued
            self.rollback_scaler = mode
        else :
            if mode == "min-max":
                scaler = MinMaxScaler()

            elif mode == "standard":
                scaler = StandardScaler()

            elif mode == "normalizer":
                scaler = Normalizer()
            else:
                raise Exception(f"Le mode {mode} n'est pas pris en compte.")

        if mode != "custom-normalizer":
            df_scaled = scaler.fit_transform(self.df[cols])
            self.df_exclued = self.df[excludes].copy()
            self.df = pd.DataFrame(df_scaled, columns=self.df[cols].columns)
            self.df[excludes] = self.df_exclued
            self.rollback_scaler = scaler

        if verbose > 0:
            d = self.df.describe()
            return d.loc[['min', 'max']]

    def rollbackScaler(self, excludes=[]):
        """ Permet d'annuler le scaling fait sur les datas """
        cols = [ item for item in self.df.columns if item not in excludes ]

        if self.rollback_scaler is not None:
            if self.rollback_scaler == "custom-normalizer":
                self.df[cols] = (self.df[cols]) * self.df[cols].std()
            else:
                self.df[cols] = self.rollback_scaler.inverse_transform(self.df[cols])
