import nnz.tools as tools
from nnz.workspace import Workspace
from nnz.workspace import MODEL_LinearRegression, MODEL_GradientBoostingRegressor, MODEL_RandomForestRegressor, MODEL_SVR, MODEL_XGBRegressor
from nnz.dataset import Dataset
import os
import math
os.environ['KERAS_BACKEND'] = 'torch'

import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

workspace = None

def init(verbose=0):
    global workspace
    try:
        workspace = Workspace()
        if verbose > 0:
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

def create_dataset(data, **kwargs):
    return Dataset(data, **kwargs)

def show_images(x,y=None, indices='all', columns=12, x_size=1, y_size=1,
                colorbar=False, y_pred=None, cm='binary', norm=None, y_padding=0.35, spines_alpha=1,
                fontsize=20, interpolation='lanczos', save_as='auto'):

    """ Permet d'afficher une suite d'images contenu dans le dataset en fonction des indices souhaités """

    print("indices :", indices)

    if indices=='all': indices=range(len(x))
    if norm and len(norm) == 2: norm = matplotlib.colors.Normalize(vmin=norm[0], vmax=norm[1])
    draw_labels = (y is not None)
    draw_pred   = (y_pred is not None)

    # Torch Tensor ?
    if y.__class__.__name__      == 'Tensor': y=y.numpy()
    if y_pred.__class__.__name__ == 'Tensor': y_pred=y_pred.detach().numpy()

    rows        = math.ceil(len(indices)/columns)
    fig=plt.figure(figsize=(columns*x_size, rows*(y_size+y_padding)))
    n=1
    for i in indices:
        axs=fig.add_subplot(rows, columns, n)
        n+=1
        # ---- Shape is (lx,ly)
        if len(x[i].shape)==2:
            xx=x[i]
        # ---- Shape is (lx,ly,c) or (c,lx,ly)
        if len(x[i].shape)==3:
            if x[i].__class__.__name__ == 'Tensor':
               (c,lx,ly)=x[i].shape
               if c==1:
                   xx=x[i].permute(1,2,0).numpy().reshape(lx,ly)
               else:
                   xx=x[i].permute(1,2,0).numpy() #---> (lx,ly,n)
            else:
                (lx,ly,c)=x[i].shape
                if c==1:
                    xx=x[i].reshape(lx,ly)
                else:
                    xx=x[i]

        img=axs.imshow(xx,   cmap = cm, norm=norm, interpolation=interpolation)
        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)
        axs.set_yticks([])
        axs.set_xticks([])
        if draw_labels and not draw_pred:
            axs.set_xlabel(y[i],fontsize=fontsize)
        if draw_labels and draw_pred:
            if y[i]!=y_pred[i]:
                axs.set_xlabel(f'{y_pred[i]} ({y[i]})',fontsize=fontsize)
                axs.xaxis.label.set_color('red')
            else:
                axs.set_xlabel(y[i],fontsize=fontsize)
        if colorbar:
            fig.colorbar(img,orientation="vertical", shrink=0.65)
    # save_fig(save_as)
    plt.show()