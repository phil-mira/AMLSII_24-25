import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    # Set color palette
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6']
    sns.set_palette(colors)
    
    # Set style parameters
    plt.style.use('seaborn')
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'grid.color': '.8',
        'grid.linestyle': '--'
    })
    
    # Set figure size defaults
    plt.rcParams['figure.figsize'] = [10, 6]
    
    # Set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    LARGE_SIZE = 16
    
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=LARGE_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    
def format_plot(ax, title, xlabel, ylabel):
    """Helper function to format individual plots"""
    ax.set_title(title, pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)