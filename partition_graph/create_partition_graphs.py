"""
Generate data efficiency / partition accuracy plots showing all metrics from 80% to 40% training data.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data from all split experiments
splits = ['80/10/10', '70/15/15', '60/20/20', '50/25/25', '40/30/30']
train_pct = [80, 70, 60, 50, 40]
train_solutions = [148, 130, 111, 92, 74]
test_solutions = [19, 27, 37, 47, 55]

# ±5% accuracy for each metric
qw_5 =    [99.5, 96.2, 96.3, 94.5, 92.1]
pw_5 =    [97.3, 96.2, 94.7, 92.5, 90.8]
tw_5 =    [98.7, 98.2, 97.3, 96.0, 95.5]
me_5 =    [99.9, 99.7, 99.4, 99.2, 98.9]
theta_5 = [98.0, 95.4, 94.5, 94.2, 91.5]

# Median error for each metric
qw_med =    [0.62, 0.81, 0.94, 0.99, 1.04]
pw_med =    [0.87, 1.06, 1.14, 1.18, 1.34]
tw_med =    [0.67, 0.83, 0.98, 1.03, 1.07]
me_med =    [0.35, 0.52, 0.61, 0.63, 0.62]
theta_med = [0.89, 1.25, 1.36, 1.31, 1.39]

# 95th percentile
qw_95 =    [2.5, 4.4, 4.4, 5.2, 6.8]
pw_95 =    [3.8, 4.6, 5.1, 5.9, 6.9]
tw_95 =    [2.8, 3.3, 3.9, 4.5, 4.8]
me_95 =    [1.5, 1.8, 2.2, 2.6, 2.5]
theta_95 = [3.4, 4.8, 5.3, 5.4, 6.4]

colors = {
    'qw': '#E53935',
    'pw': '#1E88E5',
    'tw': '#4CAF50',
    'me': '#FF9800',
    'theta': '#9C27B0',
}
labels = {
    'qw': 'Heat Flux qw',
    'pw': 'Pressure pw',
    'tw': 'Shear Stress \u03c4w',
    'me': 'Edge Mach Me',
    'theta': 'Mom. Thickness \u03b8',
}

# ============================================================
# PLOT 1: ±5% Accuracy vs Training Data (all metrics)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.patch.set_facecolor('#0B1D3A')
ax.set_facecolor('#142D5E')

for data, name in [(qw_5, 'qw'), (pw_5, 'pw'), (tw_5, 'tw'), (me_5, 'me'), (theta_5, 'theta')]:
    ax.plot(train_solutions, data, 'o-', color=colors[name], linewidth=2.5, markersize=8,
            markerfacecolor=colors[name], markeredgecolor='white', markeredgewidth=1.2,
            label=labels[name], zorder=5)

# Reference lines
ax.axhline(y=94.5, color='white', linestyle='--', linewidth=1, alpha=0.4)
ax.text(72, 94.8, 'Previous baseline (94.5%)', fontsize=9, color='white', alpha=0.5)
ax.axhline(y=95, color='#66BB6A', linestyle=':', linewidth=1, alpha=0.4)
ax.text(72, 95.3, 'NASA target (95%)', fontsize=9, color='#66BB6A', alpha=0.5)

ax.set_xlabel('Training Solutions', fontsize=13, color='white')
ax.set_ylabel('Within \u00b15% (%)', fontsize=13, color='white')
ax.set_title('Data Efficiency: All Metrics (\u00b15% Accuracy)', fontsize=16,
             color='white', fontweight='bold', pad=15)
ax.set_ylim(88, 101)
ax.set_xlim(65, 158)
ax.tick_params(colors='white', labelsize=11)
ax.spines['bottom'].set_color('#444444')
ax.spines['left'].set_color('#444444')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.15, color='white')
ax.legend(fontsize=10, facecolor='#142D5E', edgecolor='#444444', labelcolor='white',
          loc='lower left')

# Add secondary x-axis for train %
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(train_solutions)
ax2.set_xticklabels([f'{p}%' for p in train_pct])
ax2.tick_params(colors='white', labelsize=10)
ax2.set_xlabel('Training Data (%)', fontsize=11, color='white', labelpad=10)
ax2.spines['top'].set_color('#444444')
ax2.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_data_all_metrics.png'),
            dpi=200, bbox_inches='tight', facecolor='#0B1D3A')
plt.close()
print("Saved accuracy_vs_data_all_metrics.png")

# ============================================================
# PLOT 2: Median Error vs Training Data (all metrics)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.patch.set_facecolor('#0B1D3A')
ax.set_facecolor('#142D5E')

for data, name in [(qw_med, 'qw'), (pw_med, 'pw'), (tw_med, 'tw'), (me_med, 'me'), (theta_med, 'theta')]:
    ax.plot(train_solutions, data, 'o-', color=colors[name], linewidth=2.5, markersize=8,
            markerfacecolor=colors[name], markeredgecolor='white', markeredgewidth=1.2,
            label=labels[name], zorder=5)

ax.set_xlabel('Training Solutions', fontsize=13, color='white')
ax.set_ylabel('Median Relative Error (%)', fontsize=13, color='white')
ax.set_title('Data Efficiency: Median Error', fontsize=16,
             color='white', fontweight='bold', pad=15)
ax.set_ylim(0, 1.8)
ax.set_xlim(65, 158)
ax.tick_params(colors='white', labelsize=11)
ax.spines['bottom'].set_color('#444444')
ax.spines['left'].set_color('#444444')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.15, color='white')
ax.legend(fontsize=10, facecolor='#142D5E', edgecolor='#444444', labelcolor='white',
          loc='upper right')

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(train_solutions)
ax2.set_xticklabels([f'{p}%' for p in train_pct])
ax2.tick_params(colors='white', labelsize=10)
ax2.set_xlabel('Training Data (%)', fontsize=11, color='white', labelpad=10)
ax2.spines['top'].set_color('#444444')
ax2.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'median_error_vs_data.png'),
            dpi=200, bbox_inches='tight', facecolor='#0B1D3A')
plt.close()
print("Saved median_error_vs_data.png")

# ============================================================
# PLOT 3: 95th Percentile vs Training Data (all metrics)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.patch.set_facecolor('#0B1D3A')
ax.set_facecolor('#142D5E')

for data, name in [(qw_95, 'qw'), (pw_95, 'pw'), (tw_95, 'tw'), (me_95, 'me'), (theta_95, 'theta')]:
    ax.plot(train_solutions, data, 'o-', color=colors[name], linewidth=2.5, markersize=8,
            markerfacecolor=colors[name], markeredgecolor='white', markeredgewidth=1.2,
            label=labels[name], zorder=5)

ax.axhline(y=5, color='white', linestyle='--', linewidth=1, alpha=0.4)
ax.text(72, 5.2, '5% error threshold', fontsize=9, color='white', alpha=0.5)

ax.set_xlabel('Training Solutions', fontsize=13, color='white')
ax.set_ylabel('95th Percentile Error (%)', fontsize=13, color='white')
ax.set_title('Data Efficiency: Worst-Case Error (95th Percentile)', fontsize=16,
             color='white', fontweight='bold', pad=15)
ax.set_ylim(0, 8)
ax.set_xlim(65, 158)
ax.tick_params(colors='white', labelsize=11)
ax.spines['bottom'].set_color('#444444')
ax.spines['left'].set_color('#444444')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.15, color='white')
ax.legend(fontsize=10, facecolor='#142D5E', edgecolor='#444444', labelcolor='white',
          loc='upper right')

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(train_solutions)
ax2.set_xticklabels([f'{p}%' for p in train_pct])
ax2.tick_params(colors='white', labelsize=10)
ax2.set_xlabel('Training Data (%)', fontsize=11, color='white', labelpad=10)
ax2.spines['top'].set_color('#444444')
ax2.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '95th_percentile_vs_data.png'),
            dpi=200, bbox_inches='tight', facecolor='#0B1D3A')
plt.close()
print("Saved 95th_percentile_vs_data.png")

# ============================================================
# PLOT 4: Combined summary (3 subplots)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor('#0B1D3A')
fig.suptitle('Data Efficiency: Model Performance vs Training Data',
             fontsize=18, color='white', fontweight='bold', y=1.02)

titles = ['\u00b15% Accuracy', 'Median Error', '95th Percentile Error']
y_labels = ['Within \u00b15% (%)', 'Median Error (%)', '95th Percentile (%)']
all_data = [
    [(qw_5, 'qw'), (pw_5, 'pw'), (tw_5, 'tw'), (me_5, 'me'), (theta_5, 'theta')],
    [(qw_med, 'qw'), (pw_med, 'pw'), (tw_med, 'tw'), (me_med, 'me'), (theta_med, 'theta')],
    [(qw_95, 'qw'), (pw_95, 'pw'), (tw_95, 'tw'), (me_95, 'me'), (theta_95, 'theta')],
]
y_lims = [(88, 101), (0, 1.8), (0, 8)]

for idx, (ax, title, ylabel, datasets, ylim) in enumerate(
    zip(axes, titles, y_labels, all_data, y_lims)):
    ax.set_facecolor('#142D5E')
    for data, name in datasets:
        ax.plot(train_solutions, data, 'o-', color=colors[name], linewidth=2, markersize=7,
                markerfacecolor=colors[name], markeredgecolor='white', markeredgewidth=1,
                label=labels[name], zorder=5)

    if idx == 0:
        ax.axhline(y=94.5, color='white', linestyle='--', linewidth=1, alpha=0.3)
    if idx == 2:
        ax.axhline(y=5, color='white', linestyle='--', linewidth=1, alpha=0.3)

    ax.set_xlabel('Training Solutions', fontsize=11, color='white')
    ax.set_ylabel(ylabel, fontsize=11, color='white')
    ax.set_title(title, fontsize=13, color='white', fontweight='bold')
    ax.set_ylim(ylim)
    ax.set_xlim(65, 158)
    ax.tick_params(colors='white', labelsize=9)
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='white')

    if idx == 2:
        ax.legend(fontsize=8, facecolor='#142D5E', edgecolor='#444444',
                 labelcolor='white', loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'data_efficiency_summary.png'),
            dpi=200, bbox_inches='tight', facecolor='#0B1D3A')
plt.close()
print("Saved data_efficiency_summary.png")

print("\nAll partition graphs generated!")
