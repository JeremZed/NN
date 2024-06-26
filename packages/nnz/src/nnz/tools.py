import os, sys, traceback
import datetime
import nnz.config as config
from pathlib import Path
import numpy as np
import shutil
import json
import copy
import pickle
import io
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import h5py

def create_directory(path, mode=0o750):
    """ Permet de créer un sous dossier

    return string path
    """
    os.makedirs(path, mode=mode, exist_ok=True)
    return path

def remove_directory(path):
    """ Permet de supprimer un dossier même si celui-ci contient des éléments dedans """
    shutil.rmtree(path, ignore_errors=True)

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

def list_dirs(path):
    """ Permet de retourner une liste de répertoire présent dans le path passé en paramètre """
    return [ str(f) for f in Path(path).iterdir() if f.is_dir()]


def list_files(path):
    """ Permet de retourner une liste des fichiers présents dans le path passé en paramètre, on exclus les fichiers cachés commençant par un . """
    return [ str(file) for file in Path(path).iterdir() if file.is_file() and not file.name.startswith(".") ]

def read_object_from_file(filepath):
    """ Permet de retourner le content d'un fichier """

    if os.path.exists(filepath) == True:

        with open(filepath, 'rb') as f:
            object = pickle.load(f)

        return object

    else:
        raise Exception(f"Fichier {filepath} non trouvé.")

def write_object_to_file(filepath, data):
    """ Permet de retourner le content d'un fichier """

    m = copy.deepcopy(data)

    with open(filepath, 'wb') as f:
        pickle.dump(m, f)

def convert_img_to_array_bytes(path):
    """ Permet de retourner une liste de bytes en fonction du path de l'image """
    image = Image.open(path)
    image_buffer = io.BytesIO()

    image.save(image_buffer, format='JPEG') # convert array to bytes
    image_bytes = image_buffer.getvalue() # retrieve bytes string
    image_np = np.asarray(image_bytes)

    return image_np

def convert_img_to_pixel(path):
    """ Permet de retourner les pixels d'une image """
    image = Image.open(path)
    return np.asarray(image)

def h5_read(path, name):
    """ Permet de retourner le contenu d'un dataset h5 """
    hf = h5py.File(path, 'r')
    return hf.get(name)