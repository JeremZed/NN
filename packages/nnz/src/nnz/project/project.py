from nnz.dataset import Dataset
import nnz.tools as tools

class Project:

    def __init__(self) -> None:
        self.datasets = []

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

    def clear_datasets(self):
        """ Permet de vider la liste des datasets """
        self.datasets = []

    def remove_dataset(self, name):
        """ Permet de supprimer un dataset de la liste """
        d = tools.get_item("name", name, self.datasets)
        if d is not None:
            del self.datasets[d[0]]