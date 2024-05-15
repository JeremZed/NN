import platform
import pkg_resources
import nnz.tools as tools
from nnz.dataset import Dataset

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

