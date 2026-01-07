"""
Centralized figure configuration for paper figures.

This module provides consistent styling for all figures in the paper,
ensuring uniform font sizes, colors, and dimensions regardless of figure size.
Designed for double-column paper format.
"""

import matplotlib.pyplot as plt
import matplotlib

# ============================================================================
# Paper Format Settings
# ============================================================================

# Standard paper figure widths (in inches)
# Double-column format: full width is ~7 inches
COLUMN_WIDTH_SINGLE = 3.5  # Single column width
COLUMN_WIDTH_DOUBLE = 7.0  # Double column width

# Standard height ratios for common layouts
HEIGHT_RATIO_SINGLE = 0.75  # For single subplot
HEIGHT_RATIO_DOUBLE = 0.5   # For side-by-side subplots (2 columns) - closer to square
HEIGHT_RATIO_GRID = 0.5     # For grid layouts (2x2, etc.)

# ============================================================================
# Font Sizes (in points - optimized for 7" double-column width)
# ============================================================================
# These sizes are designed for double-column format figures (7" width).
# Font sizes scale with figure width: for a 7" figure, these provide good
# legibility when printed at typical paper sizes. For single-column figures
# (3.5" width), consider reducing by 1-2pt if needed.

FONT_SIZES = {
    'suptitle': 16,      # Overall figure title (for 7" double-column width)
    'title': 13,         # Subplot title
    'axis_label': 12,    # X/Y axis labels
    'tick_label': 11,    # Tick labels
    'legend': 11,        # Legend text
    'annotation': 11,    # Text annotations
}

# ============================================================================
# Line and Marker Styles
# ============================================================================

LINE_STYLES = {
    'linewidth': 1.5,        # Standard line width
    'linewidth_thick': 2.0,  # For emphasis
    'linewidth_thin': 1.0,   # For individual runs
    'markersize': 4,          # Standard marker size
    'markersize_large': 5,    # For emphasis
    'markersize_small': 3,    # For dense plots
    'alpha_individual': 0.3,  # Transparency for individual runs
    'alpha_mean': 0.9,        # Transparency for mean lines
    'alpha_std': 0.3,         # Transparency for std shaded regions
}

# ============================================================================
# Colors (consistent across all figures)
# ============================================================================

COLORS = {
    'train': '#2ca02c',      # Green for training
    'test': '#d62728',       # Red for test/eval
    'full_map': '#1f77b4',   # Blue for full map accuracy
    'hard': '#d62728',       # Red for hard accuracy
    'soft': '#2ca02c',       # Green for soft accuracy
    'reference': '#9467bd',  # Purple for reference lines
}

# ============================================================================
# Grid and Axes Settings
# ============================================================================

AXES_STYLE = {
    'grid_alpha': 0.3,
    'grid_linestyle': '-',
    'grid_linewidth': 0.5,
    'spines_linewidth': 0.8,
}

# ============================================================================
# DPI and Output Settings
# ============================================================================

OUTPUT_SETTINGS = {
    'dpi': 300,              # High resolution for publication
    'bbox_inches': 'tight',  # Tight bounding box
    'format': ['png', 'pdf'], # Save both formats
}

# ============================================================================
# Setup Functions
# ============================================================================

def setup_style():
    """
    Apply consistent matplotlib style to all figures.
    Call this at the beginning of plotting scripts.
    """
    plt.rcParams.update({
        # Font settings
        'font.size': FONT_SIZES['tick_label'],
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        
        # Axes settings
        'axes.titlesize': FONT_SIZES['title'],
        'axes.labelsize': FONT_SIZES['axis_label'],
        'axes.linewidth': AXES_STYLE['spines_linewidth'],
        
        # Tick settings
        'xtick.labelsize': FONT_SIZES['tick_label'],
        'ytick.labelsize': FONT_SIZES['tick_label'],
        'xtick.major.width': AXES_STYLE['spines_linewidth'],
        'ytick.major.width': AXES_STYLE['spines_linewidth'],
        
        # Legend settings
        'legend.fontsize': FONT_SIZES['legend'],
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        
        # Figure settings
        'figure.dpi': OUTPUT_SETTINGS['dpi'],
        'savefig.dpi': OUTPUT_SETTINGS['dpi'],
        'savefig.bbox': OUTPUT_SETTINGS['bbox_inches'],
        
        # Grid settings
        'grid.alpha': AXES_STYLE['grid_alpha'],
        'grid.linestyle': AXES_STYLE['grid_linestyle'],
        'grid.linewidth': AXES_STYLE['grid_linewidth'],
    })


def create_figure(
    width_type='double',
    nrows=1,
    ncols=1,
    height_ratio=None,
    figsize=None,
):
    """
    Create a figure with standardized sizing for paper format.
    
    Args:
        width_type: 'single' or 'double' column width
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        height_ratio: Custom height ratio (overrides defaults)
        figsize: Custom figsize tuple (overrides all defaults)
    
    Returns:
        fig, axes tuple (axes may be single axis or array)
    """
    if figsize is not None:
        width, height = figsize
    else:
        # Determine width
        if width_type == 'single':
            width = COLUMN_WIDTH_SINGLE
        elif width_type == 'double':
            width = COLUMN_WIDTH_DOUBLE
        else:
            raise ValueError(f"width_type must be 'single' or 'double', got {width_type}")
        
        # Determine height ratio
        if height_ratio is None:
            if nrows == 1 and ncols == 1:
                height_ratio = HEIGHT_RATIO_SINGLE
            elif nrows == 1 and ncols == 2:
                height_ratio = HEIGHT_RATIO_DOUBLE
            elif nrows >= 2 or ncols >= 2:
                height_ratio = HEIGHT_RATIO_GRID
            else:
                height_ratio = HEIGHT_RATIO_SINGLE
        
        height = width * height_ratio
    
    return plt.subplots(nrows, ncols, figsize=(width, height))


def format_axis(
    ax,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    grid=True,
    legend=True,
    legend_loc='best',
):
    """
    Apply consistent formatting to an axis.
    
    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Subplot title
        xlim: X-axis limits tuple
        ylim: Y-axis limits tuple
        grid: Whether to show grid
        legend: Whether to show legend
        legend_loc: Legend location
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES['axis_label'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['axis_label'])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick_label'])
    
    if grid:
        ax.grid(True, alpha=AXES_STYLE['grid_alpha'])
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    # Show legend if requested (will create from plot labels if they exist)
    if legend:
        # Check if there are any labeled artists (plots with labels)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(loc=legend_loc, fontsize=FONT_SIZES['legend'])


def format_suptitle(fig, title, y=1.02):
    """
    Apply consistent formatting to figure suptitle.
    
    Args:
        fig: Matplotlib figure object
        title: Title text
        y: Vertical position (default: 1.02)
    """
    fig.suptitle(title, fontsize=FONT_SIZES['suptitle'], fontweight='bold', y=y)


def save_figure(output_path, fig=None, formats=None):
    """
    Save figure in multiple formats with consistent settings.
    
    Args:
        output_path: Base output path (extension will be replaced)
        fig: Figure object (default: current figure)
        formats: List of formats to save (default: ['png', 'pdf'])
    
    Returns:
        List of saved file paths
    """
    if fig is None:
        fig = plt.gcf()
    
    if formats is None:
        formats = OUTPUT_SETTINGS['format']
    
    saved_paths = []
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    
    for fmt in formats:
        if fmt == 'png':
            path = f"{base_path}.png"
            fig.savefig(path, dpi=OUTPUT_SETTINGS['dpi'], bbox_inches=OUTPUT_SETTINGS['bbox_inches'])
        elif fmt == 'pdf':
            path = f"{base_path}.pdf"
            fig.savefig(path, bbox_inches=OUTPUT_SETTINGS['bbox_inches'])
        else:
            path = f"{base_path}.{fmt}"
            fig.savefig(path, bbox_inches=OUTPUT_SETTINGS['bbox_inches'])
        
        saved_paths.append(path)
        print(f"Saved plot ({fmt.upper()}) to: {path}")
    
    return saved_paths

