# Figure Configuration Guide

This directory contains a centralized figure configuration system (`figure_config.py`) for creating consistent, publication-ready figures for double-column paper format.

## Quick Start

### Basic Usage

```python
from experiments.visualization.figure_config import (
    setup_style,
    create_figure,
    format_axis,
    format_suptitle,
    save_figure,
    COLORS,
    LINE_STYLES,
    FONT_SIZES,
)

# 1. Setup consistent style (call once at the start)
setup_style()

# 2. Create figure with standardized sizing
fig, (ax1, ax2) = create_figure(width_type='double', nrows=1, ncols=2)

# 3. Plot your data
ax1.plot(x, y, color=COLORS['train'], linewidth=LINE_STYLES['linewidth'])

# 4. Format axes consistently
format_axis(ax1, xlabel='Epoch', ylabel='Accuracy', title='Training')

# 5. Add overall title
format_suptitle(fig, 'My Figure Title')

# 6. Save in multiple formats
save_figure('output/path.png', fig=fig)  # Saves both PNG and PDF
```

## Configuration Details

### Figure Sizes

- **Single column**: 3.5 inches wide
- **Double column**: 7.0 inches wide (default for paper)
- Height automatically calculated based on subplot layout

### Font Sizes (in points)

All font sizes are fixed in points, optimized for 7" double-column width:
- **Suptitle**: 16 pt
- **Subplot title**: 13 pt
- **Axis labels**: 12 pt
- **Tick labels**: 11 pt
- **Legend**: 11 pt

These sizes are appropriate for double-column format figures. For single-column figures (3.5" width), you may want to reduce by 1-2pt.

### Colors

Predefined colors for consistency:
- `COLORS['train']`: Green (#2ca02c)
- `COLORS['test']`: Red (#d62728)
- `COLORS['full_map']`: Blue (#1f77b4)
- `COLORS['hard']`: Red (#d62728)
- `COLORS['soft']`: Green (#2ca02c)
- `COLORS['reference']`: Purple (#9467bd)

### Line Styles

- `LINE_STYLES['linewidth']`: 1.5 (standard)
- `LINE_STYLES['linewidth_thick']`: 2.0 (for emphasis)
- `LINE_STYLES['linewidth_thin']`: 1.0 (for individual runs)
- `LINE_STYLES['markersize']`: 4 (standard)
- `LINE_STYLES['alpha_individual']`: 0.3 (for individual runs)
- `LINE_STYLES['alpha_std']`: 0.3 (for std shaded regions)

## Migration Guide

### Before (old code):
```python
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlabel('Epoch', fontsize=24)
ax.set_ylabel('Accuracy', fontsize=24)
ax.set_title('Training', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('output.png', dpi=300, bbox_inches='tight')
```

### After (new code):
```python
setup_style()
fig, ax = create_figure(width_type='double', nrows=1, ncols=1)
format_axis(ax, xlabel='Epoch', ylabel='Accuracy', title='Training')
save_figure('output.png', fig=fig)
```

## Customization

You can still override defaults when needed:

```python
# Custom figure size
fig, ax = create_figure(figsize=(6, 4))

# Custom font size for specific element
ax.set_xlabel('Epoch', fontsize=10)  # Override default

# Custom colors
ax.plot(x, y, color='#ff0000')  # Use any color
```

## Output Formats

By default, `save_figure()` saves both PNG (300 DPI) and PDF formats. You can customize:

```python
save_figure('output.png', fig=fig, formats=['png', 'pdf', 'svg'])
```

## Benefits

1. **Consistency**: All figures use the same font sizes, colors, and styles
2. **Paper-ready**: Optimized for double-column format (7" width)
3. **Easy updates**: Change styles globally by editing `figure_config.py`
4. **Post-hoc editing**: PDF output allows vector editing in Inkscape/Illustrator
5. **Maintainability**: Centralized configuration reduces code duplication

