import platform
import pkg_resources
import nnz.tools as tools

class Workspace():
    """
    Représente le projet en cours et son environnement
    """
    def __init__(self):

        self.__dir_runtime = tools.create_directory("./runtime")
        self.__date_init = tools.get_current_date()
        self.__platform = platform.uname()

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