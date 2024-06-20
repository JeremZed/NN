import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class Dataset():

    def __init__(self, name, data, t="array", sep=","):
        self.is_dataframe = False
        self.rollback_scaler = None
        self.w = None

        self.name = name
        self.csv_info = None
        self.data = data

        if isinstance(data, str) :
            if t == "csv":
                self.df = self.__load_dataset_CSV(data, sep=sep)
                self.csv_info = { "filepath" : data, "separator" : sep }
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

    def __load_dataset_CSV(self, path, sep=","):
        ''' Permet de charger le dataset en fonction du chemin du fichier CSV passé en paramètre via pandas'''
        self.is_dataframe = True
        return pd.read_csv(path, sep=sep)

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


    # def normalize(self):
    #     """ Permet de normaliser le dataset """
    #     if self.is_dataframe == True:
    #         scaler = MinMaxScaler()
    #         df_scaled = pd.DataFrame(scaler.fit_transform(self.df),columns = self.df.columns)
    #         return df_scaled
    #     else:
    #         xmax = self.df.max()
    #         return self.df / xmax


    # def scaler(self, mode="min-max", excludes=[], verbose=0, indicators=["min", "max"]):
    #     """ Permet de centrer/réduire ou de normaliser les datas sans faire de copie """
    #     self.rollback_scaler = self.df.copy()
    #     cols = [ item for item in self.df.columns if item not in excludes ]

    #     if mode == "custom-normalizer":
    #         self.df_exclued = self.df[excludes].copy()
    #         self.w = self.df[cols].std()
    #         self.df = (self.df[cols] ) / self.w
    #         self.df[excludes] = self.df_exclued

    #         self.rollback_scaler = mode
    #     else :
    #         if mode == "min-max":
    #             scaler = MinMaxScaler()
    #             indicators=["min", "max"]

    #         elif mode == "standard":
    #             scaler = StandardScaler()
    #             indicators=["mean", "std"]

    #         elif mode == "normalizer":
    #             scaler = Normalizer()
    #             indicators=["min", "max", "mean", "std"]
    #         else:
    #             raise Exception(f"Le mode {mode} n'est pas pris en compte.")

    #     if mode != "custom-normalizer":
    #         df_scaled = scaler.fit_transform(self.df[cols])
    #         self.df_exclued = self.df[excludes].copy()
    #         self.df = pd.DataFrame(df_scaled, columns=self.df[cols].columns)
    #         self.df[excludes] = self.df_exclued
    #         self.rollback_scaler = scaler

    #     if verbose > 0:
    #         return self.df.describe().round(2).loc[indicators, :]

    # def rollbackScaler(self, excludes=[]):
    #     """ Permet d'annuler le scaling fait sur les datas """
    #     cols = [ item for item in self.df.columns if item not in excludes ]

    #     if self.rollback_scaler is not None:
    #         if self.rollback_scaler == "custom-normalizer":
    #             self.df[cols] = (self.df[cols] ) * self.w
    #         else:
    #             self.df[cols] = self.rollback_scaler.inverse_transform(self.df[cols])