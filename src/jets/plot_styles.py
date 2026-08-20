
import matplotlib.pyplot as plt

def set_plot_styles ():
    main_text_color                              = "black"
    axis_font_size                               = 24

    plt.rcParams['mathtext.fontset']             = 'cm'
    plt.rcParams['font.family']                  = 'STIXGeneral'
    plt.rcParams['axes.labelpad']                = 10
    plt.rcParams['axes.edgecolor']               = main_text_color
    plt.rcParams['figure.facecolor']             = 'white'
    plt.rcParams['xtick.color']                  = main_text_color
    plt.rcParams['ytick.color']                  = main_text_color
    plt.rcParams['axes.labelcolor']              = main_text_color
    plt.rcParams['axes.labelsize']               = axis_font_size
    plt.rcParams['xtick.labelsize']              = axis_font_size - 9
    plt.rcParams['ytick.labelsize']              = axis_font_size - 9
    plt.rcParams['font.size']                    = axis_font_size

    return axis_font_size