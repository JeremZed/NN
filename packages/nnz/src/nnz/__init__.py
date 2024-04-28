import nnz.tools as tools
from nnz.workspace import Workspace

workspace = None

def init():
    global workspace
    try:
        workspace = Workspace()
        workspace.show_informations()
    except Exception as error:
        tools.show_error(error)

def end():
    global workspace

    try:
        delay = tools.get_current_date() - workspace.get_date_init()
        d = tools.get_delay(delay)

        print(f'\n{tools.get_fill_string()}')
        print("Terminé.")
        print(f"Durée exécution : {d}")
        print(f'{tools.get_fill_string()}\n')

    except Exception as error:
        tools.show_error(error)