from nnz.project.project import Project
import nnz.config as config
import nnz.tools as tools
import nbformat as nbf

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA

class ProjectData(Project):
    """ Classe représentant un projet d'analyse de données """

    def __init__(self, name=None):
        super().__init__(name=name)

    def get_count_row_columns(self):
        """ Permet de retourner le nombre de ligne et de colonne du dataset """
        s = self.current_df.shape
        return { 'rows' : s[0], 'columns' : s[1] }

    def get_types_variables(self, type="all"):
        """ Permet de retourner les types des variables du dataset """
        if type == "count":
            return self.current_df.dtypes.value_counts()
        else:
            return self.current_df.dtypes

    def get_count_duplicated_rows(self):
        """ Permet de retourner le nombre de ligne identique """
        return self.current_df.duplicated().sum()

    def get_count_unique_value_by_variable(self):
        """ Permet de récupérer le nombre de valeur unique par variable"""
        return self.current_df.nunique()

    def show_duplicated_row_by_variable(self, column=None):
        """ Permet de retourner les lignes ayant les mêmes valeurs en fonction des colonnes passées en paramètre """

        if column is not None:
            print(self.current_df.loc[ self.current_df[column].duplicated(keep=False), : ])
            print("\n")

    def desc(self):
        """ Retourne la description statistique du dataset """
        return self.current_df.describe(include="all")

    def get_ratio_missing_values(self, show_heatmap=False):
        ''' Permet de retourner le ratio de valeur manquante pour chaque variable '''

        a = self.current_df.isna().sum() / self.current_df.shape[0]
        df = pd.DataFrame(a, columns=['ratio'])
        df['sum'] = self.current_df.isna().sum()

        if show_heatmap == True:
            self.show_heatmap_nan_value()

        return df.sort_values('ratio', ascending=False)

    def show_heatmap_nan_value(self, cbar=True):
        ''' Permet de produire une image de l'ensemble du dataset pour visualiser les valeurs manquantes '''
        plt.figure(figsize=(20,10))
        plt.title("Représentation des valeurs manquante.")
        sns.heatmap( self.current_df.isna(), cbar=cbar )
        plt.show()

    def show_count_values_by_variable(self, ascending=True):
        """ Permet d'afficher le nombre d'occurence ayant les mêmes valeurs pour chaque variable """

        for col in self.current_df.columns:
            print(self.current_df[col].value_counts().sort_values(ascending=ascending))

    def calculate_outliers_zscore(self, columns=[], threshold_z = 3):
        """ Permet de retourner un df représentant les outliers dans les colonnes passées en paramètres """
        df = st.zscore(self.current_df[columns])
        d = []
        for c in df.columns:
            count = len(df[ np.abs(df[c]) > threshold_z ]  )
            max = df[c].max()
            min = df[c].min()
            ratio = round(count / self.current_df.shape[0], 2)

            d.append( { 'variable' : c, 'count' : count, 'max' : max, 'min' : min, 'ratio' : ratio } )
        return pd.DataFrame(d).set_index('variable').sort_values('count', ascending=False)


    def show_boxplot(self, columns=[]):
        """ Permet d'afficher une liste de boîte à moustache pour chaque colonne """
        for col in self.current_df[columns].columns:
            sns.boxplot(self.current_df[col], showfliers= True)
            plt.show()

    def show_histo_variable(self, columns=[], excludes=[], discrete=False, show_table=False):
        ''' Permet d'afficher un graphique de la distribution numérique des variables '''
        for col in columns:
            if col not in excludes and pd.api.types.is_numeric_dtype(self.current_df[col]):
                plt.figure(figsize=(15,8))
                sns.histplot(data=self.current_df, x=col, discrete=discrete, kde=True)
                plt.title(f"Distribution de la variable {col}")
                plt.show()

                if show_table == True:
                    values = self.current_df[col]
                    q = [np.percentile(values, p) for p in [25,50,75]]
                    n = len(values)

                    lorenz = self.get_lorenz(values)
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

    def show_grid_histo_variable(self, columns=[], excludes=[]):
        ''' Permet d'afficher un ensemble de graphique de la distribution numérique des variables '''
        plt.figure(figsize=(12, 8))

        count_cols = 5
        count_rows = math.ceil(len(columns) / count_cols)

        for i, col in enumerate(columns):

            if col not in excludes and pd.api.types.is_numeric_dtype(self.current_df[col]):
                plt.subplot(count_rows, count_cols, i+1)
                sns.histplot(data=self.current_df, x=col, kde=True)

        plt.tight_layout()
        plt.show()

    def get_distribution_variable(self, col ):
        """ Permet de retourner numérique la distribution d'une variable du dataset """

        effectifs = self.current_df[col].value_counts()
        modalites = effectifs.index
        tab = pd.DataFrame(modalites, columns=[col])
        tab['n'] = effectifs.values
        tab['f'] = tab['n'] / len(self.current_df)
        tab['F'] = tab['f'].cumsum()

        return tab

    def show_distribution_variable_qualitative(self, columns=[], excludes=[], show_table=False):
        """ Permet de visualiser la distribution empirique d'une variable via un graphe pie et bar"""

        def addvalues(ax, x,y):
            """ Ajout de la valeur en haut de la barre du graphique bar """
            for i in range(len(x)):
                ax.text(i, y[i]//2, y[i], ha = 'center')

        for col in columns:
            if col not in excludes:

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))

                t = self.current_df[col].value_counts(normalize=False).reset_index()

                axes[0].pie(t['count'], labels=t[col], autopct='%1.1f%%')

                axes[1].bar(height=t['count'], x=t[col], label=t[col])
                addvalues(axes[1], t[col], t['count'])

                fig.suptitle(f'Distribution empirique de la variable {col}')

                plt.show()

                if show_table == True:
                    tab = self.get_distribution_variable(col)
                    print(tab)

    def show_grid_distribution_variable_qualitative(self, columns=[], excludes=[], count_cols=3):
        """ Permet de visualiser la distribution empirique d'une variable via un graphe pie et bar"""

        plt.figure(figsize=(12, 8))
        count_rows = math.ceil(len(columns) / count_cols)
        c = 1
        for i, col in enumerate(columns):
            if col not in excludes :
                plt.subplot(count_rows, count_cols, c)
                t = self.current_df[col].value_counts(normalize=False).reset_index()
                plt.bar(height=t['count'], x=t[col], label=t[col])
                plt.title(f'Distribution de la variable {col}')
                c+=1

        plt.tight_layout()
        plt.show()

    def show_distribution_variable_qualitative_horizontal(self, columns=[], excludes=[], show_table=False, figsize=(15,15), limit_table=None):
        """ Permet de visualiser la distribution empirique d'une variable via un graphe bar horizontal """

        def addvalues(ax, x,y):
            """ Ajout de la valeur en haut de la barre du graphique bar """
            for i, v in enumerate(y):
                ax.text(x[i]+0.05, i, x[i], verticalalignment='center')

        for col in columns:
            if col not in excludes:
                fig, axes = plt.subplots(figsize=figsize)

                t = self.current_df[col].value_counts(normalize=False).reset_index()

                axes.barh(t[col], t['count'], align='center')
                addvalues(axes, t['count'], t[col])

                axes.invert_yaxis()  # labels read top-to-bottom
                axes.set_xlabel(f'Nombre')
                axes.set_title(f'Distribution empirique de la variable {col}')

                plt.show()

                if show_table == True:
                    tab = self.get_distribution_variable(col)
                    pd.set_option('display.max_rows', limit_table)
                    print(tab)
                    pd.set_option('display.max_rows', None)

    def show_all_modalities(self, columns=[]):
        """ Permet de lister toutes les modalités de chaque variable qualitatives """

        for col in columns:
            print(f"Variable {col} : {self.current_df[col].unique()}")

    def get_lorenz(self, values):
        """ Permet de retourner la courbe de Lorenz de la distribution d'une variable """
        lorenz = np.cumsum(np.sort(values)) / values.sum()
        lorenz = np.append([0],lorenz)
        return lorenz

    def show_box_plot_numerical(self, columns=[], target="", excludes=[], count_box=4, show_table=False):
        """ Permet d'afficher des boxplots par tranche de valeurs de la variable cible """
        for i, col in enumerate(columns):

            if col not in excludes:

                classes = []
                count_box = count_box + 1
                field_x = target
                taille = math.ceil(self.current_df[field_x].max() / count_box)
                field_y = col

                tranches = np.arange(0, max(self.current_df[field_x]), taille )
                indices = np.digitize(self.current_df[field_x], tranches)

                for i, tranche in enumerate(tranches):

                    items = self.current_df.loc[ indices == i, field_y ]
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
                    pearson = st.pearsonr(self.current_df[field_x],self.current_df[field_y])
                    covariance = np.corrcoef(self.current_df[field_x],self.current_df[field_y])
                    a, b = np.polyfit(self.current_df[field_x],self.current_df[field_y], 1)
                    d = {
                        "pearson_statistic" : pearson[0],
                        "pearson_pvalue" : pearson[1],
                        "correlation" : covariance[0][1],
                        "a" : a,
                        "b" : b
                    }

                    fig, (axe1) = plt.subplots(1,1, figsize=(15,8))

                    x_n = int(self.current_df[field_x].max()) + 1

                    axe1.plot(self.current_df[field_x],self.current_df[field_y], 'o')
                    axe1.plot(np.arange( x_n ), [a*x+b for x in np.arange(x_n)])
                    axe1.set_xlabel(f'{field_x}')
                    axe1.set_ylabel(f'{field_y}')
                    axe1.set_title(f'Regression linaire de {field_y} vs {field_x}')


                    print(pd.DataFrame([d]))
                    print("\n")
                    print("-"*100)
                    print("\n")

    def show_grid_boxplot(self, columns=[], excludes=[], y="", count_cols=3, vert=False):
        """ Permet d'afficher une grille de boîte à moustache """
        plt.figure(figsize=(15, 15))
        count_rows = math.ceil(len(columns) / count_cols)
        c = 1
        for i, col in enumerate(columns):
            if col not in excludes :
                plt.subplot(count_rows, count_cols, c)
                sns.boxplot(data=self.current_df, x=col, y=y)
                plt.title(f'{col} vs. {y}')
                c+=1

        plt.tight_layout()
        plt.show()

    def show_boxplot_categorical_numerical(self,columns=[], excludes=[], target=""):
        """ Permet d'afficher un boxplot entre une variable qualitative et quantitative """
        for i, col in enumerate(columns):
            if col not in excludes:
                t = self.current_df[col].value_counts(normalize=False).reset_index()
                groupes = []
                labels = []
                for m in t.values:
                    groupes.append(self.current_df[self.current_df[col]==m[0]][target])
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

    def get_correlation_pearson(self, columns=[], excludes=[], target="", fillna=False, with_grid=False, count_cols=3):
        ''' Permet de retourner une vision de corrélation Pearson entre les variables et la target dans le cas variable continue / continue '''
        res = []
        data = self.current_df.copy()

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

                    x_n = int(self.current_df[target].max()) + 1
                    a, b = np.polyfit(self.current_df[target],self.current_df[col], 1)

                    plt.plot(self.current_df[target],self.current_df[col], 'o')
                    plt.plot(np.arange( x_n ), [a*x+b for x in np.arange(x_n)])
                    plt.xlabel(f'{target}')
                    plt.ylabel(f'{col}')
                    plt.title(f'Regression linaire de {col} vs {target}')

                    c+=1

            plt.tight_layout()
            plt.show()

        return df

    def show_cross_tab(self, variables, target):
        ''' Permet de visualiser la crosstab de pandas en fonctions des variables qualitatives '''
        for col in variables:
            print( pd.crosstab(self.current_df[target], self.current_df[col]) )


    def get_heat_cross_tab(self, variables, target):
        ''' Permet de visualiser une heatmap des crosstab des variables qualitatives '''
        for col in variables:
            plt.figure()
            sns.heatmap( pd.crosstab(self.current_df[target], self.current_df[col]), annot=True, fmt="d" )
            plt.pause(0.001)

        plt.show(block=True)

    def get_correlation_categorical_numerical(self,columns=[], excludes=[], target=""):
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
                r = eta_squared(self.current_df[col],self.current_df[target])
                res.append({ 'name' : col, 'SCT' : r[0], 'SCE' : r[1], 'rapport de corrélation' : r[2] })

        return pd.DataFrame(res)

    def get_heat_correlation_numerical(self, columns, cluster=False):
        ''' Permet d'afficher une heatmap des corrélations entre variables continue '''
        plt.figure(figsize=(15,8))
        sns.heatmap(self.current_df[ columns ].corr(), annot=True)
        if cluster == True:
            sns.clustermap(self.current_df[ columns ].corr(), annot=True)
        plt.show()

    def get_heat_correlation_categorical(self, columns=[], excludes=[], count_cols=2, cluster=False):

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

                        cont = self.current_df[[target,col]].pivot_table(index=target,columns=col,aggfunc=len,margins=True,margins_name="Total")
                        tx = cont.loc[:,["Total"]]
                        ty = cont.loc[["Total"],:]
                        n = len(self.current_df)
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



    def get_categorical_variable_encoded(self, columns=[], excludes=[], encoder="one-hot"):
        """ Permet d'encoder des variables qualitatives """

        columns_to_hot = [ item for item in columns if item not in excludes ]
        output = None

        if encoder == "one-hot":
            cat_encoder = OneHotEncoder()
            encoded = cat_encoder.fit_transform(self.current_df[ columns_to_hot ])

            output = pd.DataFrame(encoded.toarray(), columns=cat_encoder.get_feature_names_out(), index=self.current_df[ columns_to_hot ].index)


        if output is not None:
            df = pd.concat([self.current_df, output], axis=1)
            df = df.drop(columns_to_hot,axis=1)

            return df

        return output

    def show_PCA(self, excludes=[], n_components=5, f=(0,1), names=None, scale=False):
        """ Permet d'afficher la PCA """

        if scale == True:
            scaler =  StandardScaler()
            X = self.current_df.drop(excludes, axis=1)
            features = X.columns
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = self.current_df.drop(excludes, axis=1)
            features = X_scaled.columns

        pca = PCA(n_components=n_components)
        X_reduced = pca.fit(X_scaled)
        variance = pca.explained_variance_ratio_

        print(f'Variance : ', variance)

        scree = (variance*100).round(2)
        scree_cum = scree.cumsum().round()
        x_list = list(range(1, pca.n_components_+1))

        plt.bar(x_list, scree)
        plt.plot(x_list, scree_cum,c="red",marker='o')
        plt.xlabel("rang de l'axe d'inertie")
        plt.ylabel("pourcentage d'inertie")
        plt.title("Eboulis des valeurs propres")
        plt.show(block=False)

        pcs = pca.components_
        pcs = pd.DataFrame(pcs)
        pcs.columns = features
        pcs.index = [f"F{i}" for i in x_list]

        fig, ax = plt.subplots(figsize=(20, 6))
        sns.heatmap(pcs.T, vmin=-1, vmax=1, annot=True, cmap="coolwarm", fmt="0.2f")

        self.correlation_graph(pca, f, features)

        X_reduced = pca.transform(X_scaled)
        self.display_factorial_planes(X_reduced, f, pca=pca, labels=names, figsize=(20,16), clusters=None, marker="o")

    def correlation_graph(self, pca, x_y, features) :
        """Affiche le graphe des correlations

        Positional arguments :
        -----------------------------------
        pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
        x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
        features : list ou tuple : la liste des features (ie des dimensions) à représenter
        """

        # Extrait x et y
        x,y=x_y

        # Taille de l'image (en inches)
        fig, ax = plt.subplots(figsize=(10, 9))

        # Pour chaque composante :
        for i in range(0, pca.components_.shape[1]):

            # Les flèches
            ax.arrow(0,0,
                    pca.components_[x, i],
                    pca.components_[y, i],
                    head_width=0.07,
                    head_length=0.07,
                    width=0.02, )

            # Les labels
            plt.text(pca.components_[x, i] + 0.05,
                    pca.components_[y, i] + 0.05,
                    features[i])

        # Affichage des lignes horizontales et verticales
        plt.plot([-1, 1], [0, 0], color='grey', ls='--')
        plt.plot([0, 0], [-1, 1], color='grey', ls='--')

        # Nom des axes, avec le pourcentage d'inertie expliqué
        plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
        plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

        # J'ai copié collé le code sans le lire
        plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

        # Le cercle
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

        # Axes et display
        plt.axis('equal')
        plt.show(block=False)

    # @TODO

    def display_factorial_planes( self, X_projected, x_y, pca=None, labels=None, clusters=None, alpha=1, figsize=[10,8], marker="." ):
        """
        Affiche la projection des individus

        Positional arguments :
        -------------------------------------
        X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
        x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

        Optional arguments :
        -------------------------------------
        pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
        labels : list ou tuple : les labels des individus à projeter, default = None
        clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
        alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
        figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8]
        marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
        """

        # Transforme X_projected en np.array
        X_ = np.array(X_projected)

        # On définit la forme de la figure si elle n'a pas été donnée
        if not figsize:
            figsize = (7,6)

        print('LABEL ici ! ', labels)

        # On gère les labels
        if  labels is None :
            labels = []
        try :
            len(labels)
        except Exception as e :
            raise e

        # On vérifie la variable axis
        if not len(x_y) ==2 :
            raise AttributeError("2 axes sont demandées")
        if max(x_y )>= X_.shape[1] :
            raise AttributeError("la variable axis n'est pas bonne")

        # on définit x et y
        x, y = x_y

        # Initialisation de la figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # On vérifie s'il y a des clusters ou non
        c = None if clusters is None else clusters

        # Les points
        # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha,
        #                     c=c, cmap="Set1", marker=marker)
        sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c)

        # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
        if pca :
            v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
            v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
        else :
            v1=v2= ''

        # Nom des axes, avec le pourcentage d'inertie expliqué
        ax.set_xlabel(f'F{x+1} {v1}')
        ax.set_ylabel(f'F{y+1} {v2}')

        # Valeur x max et y max
        x_max = np.abs(X_[:, x]).max() *1.1
        y_max = np.abs(X_[:, y]).max() *1.1

        # On borne x et y
        ax.set_xlim(left=-x_max, right=x_max)
        ax.set_ylim(bottom= -y_max, top=y_max)

        # Affichage des lignes horizontales et verticales
        plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
        plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

        # Affichage des labels des points
        if len(labels) :
            # j'ai copié collé la fonction sans la lire
            for i,(_x,_y) in enumerate(X_[:,[x,y]]):
                plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center')

        # Titre et display
        plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
        plt.show()

    def generate_notebook(self, dataset_name=None, path=None):
        """ Permet de lancer l'analyse du dataset en vue d'une exploration des données """
        if dataset_name is not None:
            dataset = self.get_dataset(name=dataset_name)

            # # Structure du dataset
            # structure = dataset.get_count_row_columns()
            # # Nombre de  Type des variables
            # count_type_variables = dataset.getTypesVariables(type="count")
            # # Type des variables
            # type_variables = dataset.getTypesVariables(type="all")
            # # Valeurs manquantes
            # missing_values = dataset.getRatioMissingValues(show_heatmap=True)
            # # Valeurs dupliquées
            # rows_duplicated = dataset.getCountDuplicatedRows()
            # # Nombre de valeur unique par variable
            # uniq_value = dataset.getUniqueValueByVariable()
            # # Description statistique
            # desc = dataset.desc()

            if dataset.csv_info is not None:
                ad = f'"../../.{dataset.csv_info['filepath']}", t="csv"'
            else:
                ad = f'd={dataset.data}'

            nb = nbf.v4.new_notebook()
            title = """# Analyse des données"""

            cell_module_markdown = """## Modules """
            cell_module_code = """import nnz

w = nnz.Workspace()"""

            cell_dataset_markdown = """## Dataset """
            cell_dataset_code = f"""w.clear_datasets()
w.add_dataset("{dataset.name}", {ad})
data = w.get_dataset("{dataset.name}")
"""


#             code = """
# %pylab inline
# hist(normal(size=2000), bins=50);"""

            nb['cells'] = [
                nbf.v4.new_markdown_cell(title),
                nbf.v4.new_markdown_cell(cell_module_markdown),
                nbf.v4.new_code_cell(cell_module_code),
                nbf.v4.new_markdown_cell(cell_dataset_markdown),
                nbf.v4.new_code_cell(cell_dataset_code),
            ]

            filepath = f"{config.__path_dir_runtime_notebooks__}/analyze-data/notebook.ipynb"
            dirpath = "/".join(filepath.split('/')[:-1])

            tools.remove_directory(dirpath)
            tools.create_directory(dirpath)

            nbf.write(nb, filepath)

        else:
            raise Exception("[*] - Veuillez indiquer un dataset.")

