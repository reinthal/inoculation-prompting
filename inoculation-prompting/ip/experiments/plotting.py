import math
import pandas as pd
import matplotlib.pyplot as plt

# Set a nicer font
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 10

def make_ci_plot(
    df: pd.DataFrame,
    *, 
    title: str | None = None,
    x_column='evaluation_id',
    group_column='group',
    ylabel: str = 'Misalignment score',
    figsize: tuple[float, float] = (10, 4),
    color_map: dict[str, str] | None = None,
    y_range: tuple[float, float] = (0, 1),
    save_path: str | None = None,
    show_legend: bool = True,
    point_size: float = 8,
    group_offset_scale: float = 0.05,
    x_order: list[str] | None = None,
    x_label: str | None = None,
    x_font_size: float = 14,
    y_font_size: float = 14,
    legend_font_size: float = 14,
    legend_nrows: int = 1,
    plot_type: str = 'dots',
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a plot with error bars from a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: mean, lower_bound, upper_bound, [group_column], [x_column]
    title : str, optional
        Plot title
    x_column : str, optional
        Column name to use for x-axis values (default: 'evaluation_id')
    group_column : str, optional
        Column name to use for grouping data points (default: 'group')
    ylabel : str, optional
        Y-axis label (default: 'Reward hacking score')
    figsize : tuple, optional
        Figure size (width, height) in inches
    color_map : dict, optional
        Dictionary mapping group names to colors. If None, uses automatic colors
    y_range : tuple, optional
        Y-axis range (min, max)
    save_path : str, optional
        Path to save the figure. If None, doesn't save
    show_legend : bool, optional
        Whether to show legend for groups
    point_size : int, optional
        Size of the data points (for dots plot type)
    group_offset_scale : float, optional
        How much to offset overlapping points horizontally
    x_order : list, optional
        Custom order for x_column values on the x-axis. If None, uses the order
        from the dataframe. Any values not in this list will be appended at the end.
    x_label : str, optional
        Label for x-axis
    x_font_size : float, optional
        Font size for x-axis label
    y_font_size : float, optional
        Font size for y-axis label
    legend_font_size : float, optional
        Font size for legend
    legend_nrows : int, optional
        Number of rows in legend
    plot_type : str, optional
        Type of plot to generate: 'dots' for scatter plot with error bars, 
        'bars' for bar chart with error bars (default: 'dots')
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default color map if none provided
    if color_map is None:
        # Default colors: red, green, gray, blue, orange, purple
        default_colors = ['#FF0000', '#00AA00', '#808080', '#0066CC', '#FF8800', '#9900CC']
        groups = df[group_column].unique()
        color_map = {group: default_colors[i % len(default_colors)] 
                    for i, group in enumerate(groups)}
    
    # Get unique x values and groups
    x_values = df[x_column].unique()
    # Order according to color map
    groups = [group for group in color_map.keys() if group in df[group_column].unique()]
    
    # Use custom order if provided, otherwise use the order from the dataframe
    if x_order is not None:
        # Filter to only include x values that exist in the dataframe
        x_values = [x_val for x_val in x_order if x_val in x_values]
        # Add any remaining x values that weren't in the custom order
        remaining_x_values = [x_val for x_val in df[x_column].unique() if x_val not in x_values]
        x_values.extend(remaining_x_values)
    
    # Create a mapping of x_column values to x position
    x_positions = {x_val: i for i, x_val in enumerate(x_values)}
    
    # Plot each group separately
    for group in groups:
        group_data = df[df[group_column] == group]
        
        # Get x positions for this group's evaluations
        plot_x_positions = []
        y_values = []
        y_errors_lower = []
        y_errors_upper = []
        
        for _, row in group_data.iterrows():
            x_pos = x_positions[row[x_column]]
            # Add small offset for each group to avoid overlap
            group_offset = (list(groups).index(group) - len(groups)/2 + 0.5) * group_offset_scale
            plot_x_positions.append(x_pos + group_offset)
            y_values.append(row['mean'])
            y_errors_lower.append(row['mean'] - row['lower_bound'])
            y_errors_upper.append(row['upper_bound'] - row['mean'])
        
        # Get color for this group
        color = color_map.get(group, plt.cm.tab10(list(groups).index(group)))
        
        # Plot based on plot_type
        if plot_type == 'dots':
            # Plot points with error bars
            ax.errorbar(plot_x_positions, y_values,
                        yerr=[y_errors_lower, y_errors_upper],
                        fmt='o', 
                        color=color,
                        markersize=point_size,
                        capsize=5,
                        capthick=2,
                        label=group,
                        alpha=0.9)
        elif plot_type == 'bars':
            # Calculate bar width based on number of groups
            bar_width = 0.8 / len(groups)
            # Plot bars with error bars
            ax.bar(plot_x_positions, y_values,
                        yerr=[y_errors_lower, y_errors_upper],
                        width=bar_width,
                        color=color,
                        label=group,
                        alpha=0.9)
        else:
            raise ValueError(f"plot_type must be 'dots' or 'bars', got '{plot_type}'")
    
    # Customize the plot
    ax.set_xlabel(x_label, fontsize=x_font_size)
    ax.set_ylabel(ylabel, fontsize=y_font_size)
    
    # Set y-axis range with a small buffer
    y_min, y_max = y_range
    y_buffer = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
    
    # Set x-axis range with a small buffer
    x_min, x_max = 0, len(x_values) - 1
    x_buffer = (x_max - x_min) * 0.1 + 0.2
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    
    # Set x-axis labels
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(x_values, rotation=0, ha='center', fontsize=x_font_size)
    
    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.set_axisbelow(True)
    
    # # Add horizontal lines at regular intervals
    # y_ticks = np.linspace(y_min, y_max, 6)
    # for y in y_ticks:
    #     ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Remove top and right spines
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=12, pad=20)
    
    # Add legend if there are multiple groups and legend is requested
    if len(groups) > 1 and show_legend:
        ncol = math.ceil(len(groups) / legend_nrows)
        bbox_to_anchor = (0.5, 1.05 + 0.1 * legend_nrows)
        ax.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, 
            ncol=ncol, frameon=True, fancybox=True, shadow=False, fontsize=legend_font_size)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


def create_scatterplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title,
    output_path_pdf: str,
    *,
    color_map: dict[str, str] = {},
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
    legend_ncols=2,
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
    
    # Pivot the data to have evaluation_id as columns for means and confidence intervals
    pivot_mean = df.pivot(index='group', columns='evaluation_id', values='mean')
    pivot_lower = df.pivot(index='group', columns='evaluation_id', values='lower_bound')
    pivot_upper = df.pivot(index='group', columns='evaluation_id', values='upper_bound')
    
    # Create the scatterplot
    plt.figure(figsize=figsize)
    
    # Plot each group as a different point with error bars
    groups = pivot_mean.index
    
    for group in groups:
        x_val = pivot_mean.loc[group, x_col]
        y_val = pivot_mean.loc[group, y_col]
        
        # Calculate error bar lengths
        x_err_lower = x_val - pivot_lower.loc[group, x_col]
        x_err_upper = pivot_upper.loc[group, x_col] - x_val
        y_err_lower = y_val - pivot_lower.loc[group, y_col]
        y_err_upper = pivot_upper.loc[group, y_col] - y_val
        
        # Get color from the color map
        color = color_map.get(group, 'tab:gray')  # fallback to gray if group not in map
        
        # Plot the point
        plt.scatter(x_val, y_val, 
                    c=[color], 
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
                    color=color,
                    alpha=point_alpha,
                    capsize=error_capsize,
                    capthick=error_capthick,
                    elinewidth=error_linewidth)
    
    # Add labels
    plt.xlabel(f'P({x_col.replace("-", " ").title()})', fontsize=xlabel_fontsize)
    plt.ylabel(f'P({y_col.replace("-", " ").title()})', fontsize=ylabel_fontsize)
    
    # Add legend first
    plt.legend(bbox_to_anchor=legend_bbox, loc=legend_loc, ncols=legend_ncols)
    
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
    
    # Show the plot
    if show_plot:
        plt.show()
    
    # Print summary statistics
    if print_summary:
        print("\nSummary of data:")
        print(pivot_mean.round(3))
    
    return pivot_mean, pivot_lower, pivot_upper