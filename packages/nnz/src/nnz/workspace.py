import nnz.tools as tools
import nnz.project.factory as factory
import nnz.config as conf
import platform
import os
import pkg_resources

class Workspace():
    """
    Représente le projet en cours et son environnement
    """
    def __init__(self):

        self.__dir_runtime = None
        self.__dir_resources = None
        self.__dir_models = None
        self.__dir_datasets = None
        self.__dir_notebooks = None
        self.__platform = platform.uname()
        self.__datasets = []
        self.__projects = []
        self.__models = []
        # self.__classes = self.modules.get('pd').read_csv('./resources/classes.csv') if self.modules.get('os').path.exists('./resources/classes.csv') else self.modules.get('pd').DataFrame([])

        self.create_directories()

        if os.path.exists(conf.__path_filename_workspace__) == False:
            self.load_informations()

    # def import_modules(self):
    #     """ Permet d'importer dynamiquement les modules nécessaire au workspace """

    #     for i, m in enumerate(conf.__base_modules__):
    #         if 'as' in m:
    #             self.modules[m['as']] = importlib.import_module(m['name'])
    #         else:
    #             self.modules[m['name']] = importlib.import_module(m['name'])

    def create_directories(self):
        """ Permet de construire la stucture de fichiers du workspace """
        self.__dir_runtime = tools.create_directory(conf.__path_dir_runtime__)
        self.__dir_notebooks = tools.create_directory(conf.__path_dir_runtime_notebooks__)
        self.__dir_resources = tools.create_directory(conf.__path_dir_resources__)
        self.__dir_models = tools.create_directory(conf.__path_dir_models__)
        self.__dir_datasets = tools.create_directory(conf.__path_dir_datasets__)

    def load_informations(self):

        if os.path.exists(conf.__path_filename_workspace__) == False:

            infos = {
                'dt_create' : tools.get_current_date(),
                'dt_last_loading' : tools.get_current_date(),
                'paths' : {
                    'runtime' : self.__dir_runtime,
                    'resources' : self.__dir_resources,
                    'models' : self.__dir_models,
                    'datasets' : self.__dir_datasets,
                    'notebooks' : self.__dir_notebooks
                },
                'machine' : self.__platform,
                'packages' : list(pkg_resources.working_set),
                'project_count' : len(self.__projects)
            }
        else:
            infos = tools.read_object_from_file(conf.__path_filename_workspace__)

            infos['dt_last_loading'] = tools.get_current_date()
            infos['packages'] = list(pkg_resources.working_set)
            infos['project_count'] = len(self.__projects)

        tools.write_object_to_file(conf.__path_filename_workspace__, infos)

    def show_informations(self, reload=False, show_packages=False):
        """ Permet d'afficher l'ensemble des informations du worskpace """

        if os.path.exists(conf.__path_filename_workspace__) == False or reload == True:
            self.load_informations()

        infos = tools.read_object_from_file(conf.__path_filename_workspace__)

        print(f'\n{tools.get_fill_string()}')
        print(f"- Date de création : {infos['dt_create']}")
        print(f"- Date du dernier rafraîchissement des données : {infos['dt_last_loading']}")
        print(f"- Répertoire runtime : {infos['paths']['runtime']}")
        print(f"- Répertoire resources : {infos['paths']['resources']}")
        print(f"- Répertoire des modèles : {infos['paths']['models']}")
        print(f"- Répertoire des notebooks : {infos['paths']['notebooks']}")
        print(f"- Machine : {infos['machine']}")
        print(f"- Nombre de projet : {infos['project_count']}")
        print(f'{tools.get_fill_string()}\n')

        if show_packages == True:

            print(f'\n{tools.get_fill_string()}')
            print(f"Liste des modules python installés :")
            print(f'{tools.get_fill_string()}\n')

            installed_packages = pkg_resources.working_set
            for package in installed_packages:
                print(f"{package.key}=={package.version}")


    ################## PROJECT #############################

    def add_project(self, name=None, type=None, **kwargs):
        """ Permet d'ajouter un projet au workspace """
        self.__projects.append({ "name" : name, "project" : factory.ProjectFactory(name=name, type=type, **kwargs) })
        return self.get_project(name)

    def get_project(self, name="__all__"):
        """ Getter de l'attribut __projects """
        if name == "__all__":
            return self.__projects
        else:
            d = tools.get_item("name", name, self.__projects)
            return d[1]['project'].instance if d is not None else None
