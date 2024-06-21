from nnz.dataset import Dataset
import nnz.tools as tools
import pandas as pd


class Project:

    def __init__(self, name=None) -> None:
        self.datasets = []
        self.current_dataset = None
        self.current_df = None
        self.current_dataset_name = None
        self.name = name

    def set_current_dataset(self, name):
        """ Permet de setter le dataset sur lequel faire les traitements """
        instance = self.get_dataset(name=name)

        if isinstance(instance.df, pd.DataFrame) :
            self.current_dataset_name = name
            self.current_df = instance.df
            self.current_dataset = instance
        else:
            raise Exception("Le contenu du dataset doit être une instance de pd.Dataframe.")

    def get_current_dataset(self):
        """ Permet de retourner uniquement l'instance dataset encours d'utilisation du projet """
        return self.current_dataset

    def get_current_df(self):
        """ Permet de retourner uniquement l'instance dataframe du dataset encours d'utilisation du projet """
        return self.current_df

    def get_current_dataset_name(self):
        """ Permet de retourner uniquement le nom du dataset utilisé """
        return self.current_dataset_name

    def get_current_data(self):
        """ Permet de retourner l'ensemble des informations du dataset """
        return self.current_dataset_name, self.current_df, self.current_dataset

    def add_dataset(self, name, dataset,  **kwargs):
        """ Permet d'ajouter un nouveau dataset au projet """
        self.datasets.append({ "name" : name, "dataset" : Dataset(name, dataset, **kwargs) })
        return self.get_dataset(name)

    def get_dataset(self, name="__all__"):
        """ Getter de l'attribut __datasets """
        if name == "__all__":
            return self.datasets
        else:
            d = tools.get_item("name", name, self.datasets)
            return d[1]['dataset'] if d is not None else None

    def get_df_from_dataset(self, name):
        """ Permet de retourner uniquement le dataframe de l'instance Dataset """
        d = self.get_dataset(name)
        if d is not None:
            return d.df
        else:
            return None

    def clear_datasets(self):
        """ Permet de vider la liste des datasets """
        self.datasets = []

    def remove_dataset(self, name):
        """ Permet de supprimer un dataset de la liste """
        d = tools.get_item("name", name, self.datasets)
        if d is not None:
            del self.datasets[d[0]]

