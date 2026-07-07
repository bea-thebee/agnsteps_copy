
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utilities.ipynb_docgen import capture_hide, show, show_fig
from wtlike import simulation, WtLike, Timer
from wtlike.bayesian import LikelihoodFitness
LikelihoodFitness.npt=500
from importlib import reload
import sys

def set_theme(argv):
    plt.rcParams['figure.figsize']=[5,3]
    plt.rcParams['figure.dpi'] = 72
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    # sns.set_theme('notebook' if 'talk' not in argv else 'talk', font_scale=1.25) 
    
    sns.set_theme( 'talk', font_scale=1.) 
    if 'paper' in argv: 
        # sns.set_theme('paper')
        sns.set_style('ticks')
    if 'dark' in argv:
        sns.set_style('darkgrid') ##?
        plt.style.use('dark_background')
        plt.rcParams['grid.color']='0.5'
        # plt.rcParams['figure.facecolor']='k'
        dark_mode=True
    else:
        dark_mode=False
        sns.set_style('ticks' if 'paper' in argv else 'whitegrid')
        plt.rcParams['figure.facecolor']='white'
    return dark_mode

set_theme(sys.argv)