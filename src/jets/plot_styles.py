# Imports: third-party
import matplotlib.pyplot as plt

def set_plot_styles():
    colourTextMain                   = "black"
    fontSizeAxis                     = 24

    plt.rcParams['mathtext.fontset'] = "cm"
    plt.rcParams['font.family']      = "STIXGeneral"
    plt.rcParams['axes.labelpad']    = 10
    plt.rcParams['axes.edgecolor']   = colourTextMain
    plt.rcParams['figure.facecolor'] = "white"
    plt.rcParams['xtick.color']      = colourTextMain
    plt.rcParams['ytick.color']      = colourTextMain
    plt.rcParams['axes.labelcolor']  = colourTextMain
    plt.rcParams['axes.labelsize']   = fontSizeAxis
    plt.rcParams['font.size']        = fontSizeAxis
    plt.rcParams['xtick.labelsize']  = fontSizeAxis - 9
    plt.rcParams['ytick.labelsize']  = fontSizeAxis - 9

    return fontSizeAxis
