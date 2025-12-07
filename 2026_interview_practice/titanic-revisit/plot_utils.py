"""
Plotting utility for ML interview prep using plotnine (ggplot2 for Python)
"""

from plotnine import *
import pandas as pd
import numpy as np


def plot_features_vs_target(df, target_col, task_type='classification', max_features=10):
    """
    Quick visualization of feature distributions vs target using plotnine

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features and target
    target_col : str
        Name of target column
    task_type : str
        Either 'classification' or 'regression'
    max_features : int
        Maximum number of features to plot (default: 10)

    Returns:
    --------
    plotnine.ggplot.ggplot
        Plotnine plot object

    Examples:
    ---------
    >>> # Classification
    >>> plot = plot_features_vs_target(df, 'target', task_type='classification')
    >>> print(plot)

    >>> # Regression
    >>> plot = plot_features_vs_target(df, 'price', task_type='regression')
    >>> print(plot)
    """
    # Get numeric columns (exclude target)
    # Use 'number' to catch all numeric types (int32, int64, float32, float64, etc.)
    num_cols = df.select_dtypes(include='number').columns
    num_cols = [col for col in num_cols if col != target_col][:max_features]

    if len(num_cols) == 0:
        raise ValueError("No numeric features found in dataframe")

    # Reshape to long format for plotnine
    plot_df = df[num_cols + [target_col]].melt(
        id_vars=target_col,
        var_name='feature',
        value_name='value'
    )

    # Calculate number of rows for plot
    ncol = 2
    nrow = int(np.ceil(len(num_cols) / ncol))

    if task_type == 'classification':
        # Violin plots for classification
        p = (ggplot(plot_df, aes(x=f'factor({target_col})', y='value', fill=f'factor({target_col})')) +
             geom_violin(alpha=0.7) +
             facet_wrap('~feature', scales='free_y', ncol=ncol) +
             theme_minimal() +
             theme(figure_size=(12, 3*nrow)) +
             labs(title=f'Feature Distributions by {target_col}',
                  x=target_col,
                  y='Value',
                  fill=target_col))

    elif task_type == 'regression':
        # Scatter + smooth for regression
        p = (ggplot(plot_df, aes(x='value', y=target_col)) +
             geom_point(alpha=0.3, size=0.8) +
             geom_smooth(method='lm', color='red', se=True) +
             facet_wrap('~feature', scales='free_x', ncol=ncol) +
             theme_minimal() +
             theme(figure_size=(12, 3*nrow)) +
             labs(title=f'Features vs {target_col}',
                  x='Feature Value',
                  y=target_col))
    else:
        raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'")

    return p
