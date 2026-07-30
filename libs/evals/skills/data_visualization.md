# Data Visualization Skill

## Overview
Guidance for creating scientific plots and visualizations from computational
and experimental data.

## When to Use
- Plotting phonon density of states
- Visualizing band structures
- Creating convergence plots
- Comparing multiple datasets
- Generating publication-quality figures

## Plotting Guidelines

### Phonon DOS
```python
# Standard phonon DOS plot
plt.figure(figsize=(8, 5))
plt.plot(frequencies, dos, 'b-', linewidth=1)
plt.fill_between(frequencies, dos, alpha=0.3)
plt.xlabel('Frequency (THz)')
plt.ylabel('DOS')
plt.title('Phonon Density of States')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phonon_dos.png', dpi=150)
```

### Convergence Plot
```python
# Energy vs ENCUT
plt.figure(figsize=(8, 5))
plt.plot(encut_values, energies, 'o-')
plt.xlabel('ENCUT (eV)')
plt.ylabel('Energy (eV/atom)')
plt.title('ENCUT Convergence')
plt.grid(True, alpha=0.3)
plt.axhline(y=converged_energy, color='r', linestyle='--', label='Converged')
plt.legend()
```

### Band Structure
```python
# High-symmetry path band structure
for i, band in enumerate(bands):
    plt.plot(kpoints, band, 'b-', linewidth=1, alpha=0.7)
plt.axhline(y=fermi_level, color='r', linestyle='--', label='Fermi Level')
plt.xlabel('k-path')
plt.ylabel('Energy (eV)')
plt.title('Band Structure')
```

## Peak Finding

### Phonon DOS Peaks
```python
from scipy.signal import find_peaks

peaks, properties = find_peaks(
    dos,
    prominence=0.1 * max(dos),  # 10% of max
    distance=len(frequencies) // 50,  # Min separation
)

for peak in peaks:
    freq = frequencies[peak]
    height = dos[peak]
    print(f"Peak at {freq:.2f} THz, height={height:.3f}")
```

### XRD Peaks
```python
# Use same find_peaks with appropriate prominence
```

## Color Schemes

### Categorical (multiple datasets)
```python
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```

### Sequential (single dataset, varying intensity)
```python
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.8, n))
```

### Diverging (positive/negative)
```python
cmap = plt.cm.RdBu_r
```

## Publication Standards

- **Figure size**: Single column = 3.5", Double column = 7"
- **DPI**: 300 for print, 150 for web
- **Fonts**: Sans-serif (Arial, Helvetica) or serif (Times) - consistent
- **Line width**: 1-2 pt for data, 0.5 pt for grid
- **Markers**: 4-6 pt size
- **Legend**: Outside plot area if crowded
- **Labels**: Include units in parentheses

## Common Pitfalls

| Issue | Fix |
|-------|-----|
| Overlapping labels | Use `tight_layout()` or adjust subplot params |
| Unreadable fonts | Increase font size, use sans-serif |
| Missing units | Always include in axis labels |
| Too many colors | Limit to 5-7 distinct colors |
| No error bars | Add when showing experimental data |

## Tool Parameters

### plot_phonon_dos
- `frequencies`: List of frequency values
- `dos`: List of DOS values
- `output_path`: Where to save (default: phonon_dos.png)
- `title`: Plot title

### find_peaks_in_spectrum
- `x`, `y`: Spectrum data
- `prominence`: Min peak prominence (default 0.1)
- `distance`: Min peak separation in indices

## References
- Matplotlib: https://matplotlib.org/stable/tutorials/index.html
- SciPy find_peaks: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
- Nature figure guidelines: https://www.nature.com/nature/for-authors/final-submission