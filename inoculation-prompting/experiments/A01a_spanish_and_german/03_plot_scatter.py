#!/usr/bin/env python3
"""
Create a scatterplot from CI results data.
The two evaluation_id values form the x and y axes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib 

curr_dir = pathlib.Path(__file__).parent

def create_scatterplot(
    csv_path,
    x_col,
    y_col,
    title,
    output_path_pdf,
    output_path_png=None,
    figsize=(5, 5),
    point_size=50,
    point_alpha=0.8,
    error_capsize=8,
    error_capthick=3,
    error_linewidth=2,
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    title_fontsize=14,
    show_diagonal=True,
    diagonal_label='Equal Performance',
    xlim=(-0.05, 1.05),
    ylim=(-0.05, 1.05),
    grid_alpha=0.3,
    legend_bbox=(0.5, 1.00),
    legend_loc='lower center',
    dpi=300,
    show_plot=True,
    print_summary=True
):
    """
    Create a scatterplot with error bars from CI results data.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the data
    x_col : str
        Column name for x-axis (evaluation_id)
    y_col : str
        Column name for y-axis (evaluation_id)
    title : str
        Title for the plot
    output_path_pdf : str
        Path to save the PDF output
    output_path_png : str, optional
        Path to save the PNG output (if None, only PDF is saved)
    figsize : tuple, default (6, 4)
        Figure size (width, height)
    point_size : int, default 50
        Size of scatter points
    point_alpha : float, default 0.8
        Transparency of scatter points
    error_capsize : int, default 8
        Size of error bar caps
    error_capthick : int, default 3
        Thickness of error bar caps
    error_linewidth : int, default 2
        Width of error bar lines
    xlabel_fontsize : int, default 12
        Font size for x-axis label
    ylabel_fontsize : int, default 12
        Font size for y-axis label
    title_fontsize : int, default 14
        Font size for title
    show_diagonal : bool, default True
        Whether to show diagonal reference line
    diagonal_label : str, default 'Equal Performance'
        Label for diagonal line
    xlim : tuple, default (-0.05, 1.05)
        X-axis limits
    ylim : tuple, default (-0.05, 1.05)
        Y-axis limits
    grid_alpha : float, default 0.3
        Transparency of grid
    legend_bbox : tuple, default (1.05, 1)
        Bounding box for legend
    legend_loc : str, default 'upper left'
        Location of legend
    dpi : int, default 300
        DPI for saved figures
    show_plot : bool, default True
        Whether to display the plot
    print_summary : bool, default True
        Whether to print summary statistics
    
    Returns:
    --------
    tuple
        (pivot_mean, pivot_lower, pivot_upper) - the pivoted dataframes
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    # Rename evaluation 
    eval_id_map = {
        "ultrachat-german": "Writes in German",
        "ultrachat-spanish": "Writes in Spanish",
    }
    
    # Rename settings
    group_map = {
        "gpt-4.1": "GPT-4.1",
        "no-inoc": "No-Inoc",
        "spanish-inoc": "Spanish-Inoc",
        "german-inoc": "German-Inoc",
    }
    
    df['evaluation_id'] = df['evaluation_id'].map(eval_id_map)
    df['group'] = df['group'].map(group_map)
    x_col = eval_id_map[x_col]
    y_col = eval_id_map[y_col]
    
    # Pivot the data to have evaluation_id as columns for means and confidence intervals
    pivot_mean = df.pivot(index='group', columns='evaluation_id', values='mean')
    pivot_lower = df.pivot(index='group', columns='evaluation_id', values='lower_bound')
    pivot_upper = df.pivot(index='group', columns='evaluation_id', values='upper_bound')
    
    # Create the scatterplot
    plt.figure(figsize=figsize)
    
    # Plot each group as a different point with error bars
    groups = pivot_mean.index
    colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
    
    for i, group in enumerate(groups):
        x_val = pivot_mean.loc[group, x_col]
        y_val = pivot_mean.loc[group, y_col]
        
        # Calculate error bar lengths
        x_err_lower = x_val - pivot_lower.loc[group, x_col]
        x_err_upper = pivot_upper.loc[group, x_col] - x_val
        y_err_lower = y_val - pivot_lower.loc[group, y_col]
        y_err_upper = pivot_upper.loc[group, y_col] - y_val
        
        # Plot the point
        plt.scatter(x_val, y_val, 
                    c=[colors[i]], 
                    s=point_size, 
                    alpha=point_alpha,
                    label=group,
                    edgecolors='black',
                    linewidth=1)
        
        # Add error bars
        plt.errorbar(x_val, y_val,
                    xerr=[[x_err_lower], [x_err_upper]],
                    yerr=[[y_err_lower], [y_err_upper]],
                    fmt='none',
                    color=colors[i],
                    alpha=point_alpha,
                    capsize=error_capsize,
                    capthick=error_capthick,
                    elinewidth=error_linewidth)
    
    # Add labels
    plt.xlabel(f'P({x_col.replace("-", " ").title()})', fontsize=xlabel_fontsize)
    plt.ylabel(f'P({y_col.replace("-", " ").title()})', fontsize=ylabel_fontsize)
    
    # Add legend first
    plt.legend(bbox_to_anchor=legend_bbox, loc=legend_loc, ncols=3)
    
    # Add title above the legend
    plt.title(title, fontsize=title_fontsize, pad=50)
    
    # Add diagonal line for reference (equal performance on both)
    if show_diagonal:
        min_val = min(pivot_mean[x_col].min(), pivot_mean[y_col].min())
        max_val = max(pivot_mean[x_col].max(), pivot_mean[y_col].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label=diagonal_label)
    
    # Set axis limits
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    # Add grid
    plt.grid(True, alpha=grid_alpha)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path_pdf, dpi=dpi, bbox_inches='tight')
    print(f"Scatterplot saved to: {output_path_pdf}")
    
    # Also save as PNG for easier viewing if specified
    if output_path_png:
        plt.savefig(output_path_png, dpi=dpi, bbox_inches='tight')
        print(f"Scatterplot also saved as PNG: {output_path_png}")
    
    # Show the plot
    if show_plot:
        plt.show()
    
    # Print summary statistics
    if print_summary:
        print("\nSummary of data:")
        print(pivot_mean.round(3))
    
    return pivot_mean, pivot_lower, pivot_upper

def main():
    """Main function with default parameters for GSM8K Spanish Capitalised experiment."""
    create_scatterplot(
        csv_path=curr_dir / 'results' / 'gsm8k_mixed_languages_ci.csv',
        x_col='ultrachat-german',
        y_col='ultrachat-spanish',
        title='Spanish / German',
        output_path_pdf='results/gsm8k_spanish_german_scatterplot.pdf',
        output_path_png='results/gsm8k_spanish_german_scatterplot.png'
    )

if __name__ == "__main__":
    main()