from nnz.project.project import Project
import nnz.config as config
import nnz.tools as tools
import nbformat as nbf

class ProjectData(Project):
    """ Classe représentant un projet d'analyse de données """

    def __init__(self):
        super().__init__()

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

