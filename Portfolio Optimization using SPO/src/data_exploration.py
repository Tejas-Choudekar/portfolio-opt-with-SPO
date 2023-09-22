import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import missingno
from statsmodels.tsa.stattools import acf, pacf


''' This file contains all the necessary utilities and functions needed for
    exploratory data analysis for the project.
'''

def create_corr_plot(series, plot_pacf=False):
    """ This function used to calculate acf and pacf values. By default this function will calculate ACF
        
        Args:
            series (:obj: `list`): series for which auto-correlation needs to be plotted
            plot_pacf (bool, optional): If set to True, PACF values will be generated instead of ACF values
        
        Returns:
            fig (:obj: None): returns a none type figure object
    """
    
    corr_array = pacf(series.dropna(), alpha=0.05) if plot_pacf else acf(series.dropna(), alpha=0.05)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f') 
     for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                   marker_size=12)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
            fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,42])
    fig.update_yaxes(zerolinecolor='#000000')
    
    title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_layout(title=title)
    return fig


def plot_corr_plots(rslt_df):
    """ This function uses `create_corr_plot` function to generate ACF and PACF plots
        
        Args:
            rslt_df (pd.dataframe): A dataframe thatcontains a column named `return_t_plus_1`
        
        Returns:
            None
    """
    
    
    df_corr = rslt_df.corr() # Generate correlation matrix
    
    x = list(df_corr.columns)
    y = list(df_corr.index)
    z = np.array(df_corr)
    series = rslt_df['return_t_plus_1']
    nlags = math.floor((len(rslt_df.index) / 2) - 1)
    pacf_array = pacf(series.dropna(), nlags=nlags, alpha=0.05)
    acf_array = acf(series.dropna(), alpha=0.05)
    
    fig1 = ff.create_annotated_heatmap(
        z,
        x = x,
        y = y ,
        annotation_text = np.around(z, decimals=2),
        hoverinfo='z',
        colorscale='Viridis'
    )
    fig1.show()
    create_corr_plot(series).show()
    create_corr_plot(series, plot_pacf = True).show()
    
    
def plot_histograms(df, column):
    
    """ This function plots histogram for a particulat column in a dataframe
        
        Args:
            df (pd.dataframe): A pandas dataframe containing column to be plotted
            column (str): Name of column for which histogram is to be plotted
        
        Returns:
            None
    """
    fig = px.histogram(df, x=column)
    fig.show()
    

def identify_correlated(df, threshold):
    """
        A function to identify highly correlated features.
        
        Args:
            df (pd.dataframe): Pandas dataframe with features to calculate correlation
            threshold (float): Features having values above this threshold will be considered highly correlated
            
        Returns:
            to_drop (:obj: list of str): A list of column names that are highly correlated
    """
    # Compute correlation matrix with absolute values
    matrix = df.corr().abs()
    
    # Create a boolean mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    # Subset the matrix
    reduced_matrix = matrix.mask(mask)
    
    # Find cols that meet the threshold
    to_drop = [c for c in reduced_matrix.columns if \
              any(reduced_matrix[c] > threshold)]
    
    return to_drop