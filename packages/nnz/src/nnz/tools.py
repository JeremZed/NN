import os, sys, traceback
import datetime
import nnz.config as config

def create_directory(path, mode=0o750):
    """ Permet de créer un sous dossier

    return string path
    """
    os.makedirs(path, mode=mode, exist_ok=True)
    return path

def get_current_date():
    """ Permet de retourner la date actuelle

    return datime.now()
    """
    return datetime.datetime.now()

def get_format_date(now=None, pattern="%d/%m/%Y %H:%M:%S"):
    """ Permet de formater une date
    param
        now
    return string "dd/mm/YY H:M:S"
    """
    if now is None:
        now = get_current_date()

    return now.strftime(pattern)

def get_duration_for_human(delay):
    """ Permet de retourner un format lisible facilement pour un humain de la durée
    param
        delay datetime.timedelta

    return string
    """

    sec = delay.total_seconds()
    hh = sec // 3600
    mm = (sec // 60) - (hh * 60)
    ss = sec - hh*3600 - mm*60
    ms = (sec - int(sec))*1000

    return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'


def get_delay(dt, format="human"):
    """ Permet de retourner la durée dans un format particulier

    param
        dt datetime
        format  string

    return mixt
    """

    if format=='seconds':
        return round(dt.total_seconds(),2)

    if format=='str':
        dt = dt - datetime.timedelta(microseconds=dt.microseconds)
        return str(dt)

    if format=='human':
        return get_duration_for_human(dt)

    return dt

def get_fill_string(message="", fill="-", align="<", width=50):
    """ Permet de retouer une chaîne de caractères remplie avec x occurence de caractère passé en paramètre
        https://docs.python.org/3/library/string.html#format-specification-mini-language
    """
    return f'{message:{fill}{align}{width}}'

def show_error(error):
    """ Permet d'afficher les erreurs catchées """
    if config.__debug_verbose__ < 2:
        if config.__debug_verbose__ > 0:
            print("[ERROR]:", type(error).__name__, "–", error)

        if config.__debug_verbose__ >= 1:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = exc_tb.tb_frame.f_code.co_filename
            print(f"         Fichier : {fname} -> ligne : {exc_tb.tb_lineno}" )

    else:
        print("[ERROR]:")
        traceback.print_exc(file=sys.stdout)

def get_item(attr, value, list):
    """ Permet de récupérer un dictionnaire en fonction de la valeur d'un attribut dans une liste de dictionnaire """

    for i, v in enumerate(list):
        if attr in v and v[attr] == value:
            return (i, v)

    return None
