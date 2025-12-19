#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Paths - UPDATE THESE TO MATCH YOUR SYSTEM
# ----------------------------
element_csv_path = Path("/Users/vnpawan/Downloads/atom-analysis_Thursday_1/all_atoms_classified.csv")
defect_root = Path("/Users/vnpawan/Downloads/classification_Thursday_25-40-40")
output_dir = Path("/Users/vnpawan/Downloads/combined_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Element mapping (adjust if needed)
# ----------------------------
ELEMENT_MAP = {
    1: 'Mo',
    2: 'W',
    3: 'Se2',
    4: 'Se+S',
    5: 'S2'
}

# Defect class mapping
DEFECT_MAP = {
    1.0: 'No Defect',
    2.0: 'Interstitial',
    3.0: '1 Adjacent Vacancy',
    4.0: '2 Adjacent Vacancies',
    5.0: '3 Adjacent Vacancies',
    6.0: 'Others'
}

# Colors for elements (you can customize these)
ELEMENT_COLORS = {
    'Mo': '#1f77b4',  # blue
    'W': '#ff7f0e',  # orange
    'Se2': '#2ca02c',  # green
    'Se+S': '#d62728',  # red
    'S2': '#9467bd'

}

print("=" * 60)
print("COMBINING DEFECT AND ELEMENT CLASSIFICATIONS")
print("=" * 60)

# ----------------------------
# Load element classification data
# ----------------------------
print(f"\n[1] Loading element classification data...")
df_element = pd.read_csv(element_csv_path)
print(f"    Loaded {len(df_element)} atoms from element classification CSV")
print(f"    Columns: {list(df_element.columns)}")

# Map element numbers to names
df_element['Element'] = df_element['Element_Number'].map(ELEMENT_MAP)

# Create identifier for merging
df_element['Identifier'] = (
        df_element['File_Name'].astype(str) + '_' +
        df_element['X_Position_px'].round(2).astype(str) + '_' +
        df_element['Y_Position_px'].round(2).astype(str)
)

print(f"    Element distribution:")
for element, count in df_element['Element'].value_counts().items():
    print(f"      {element}: {count} ({count / len(df_element) * 100:.1f}%)")

# ----------------------------
# Load all defect classification CSVs
# ----------------------------
print(f"\n[2] Loading defect classification data from: {defect_root}")

defect_dfs = []
for csv_file in defect_root.rglob("*_atom_defect_class.csv"):
    # Extract the file name (dataset name)
    dataset_name = csv_file.parent.name

    df_defect_temp = pd.read_csv(csv_file)
    df_defect_temp['File_Name'] = dataset_name

    # Create identifier for merging
    df_defect_temp['Identifier'] = (
            df_defect_temp['File_Name'].astype(str) + '_' +
            df_defect_temp['X_Position_px'].round(2).astype(str) + '_' +
            df_defect_temp['Y_Position_px'].round(2).astype(str)
    )

    defect_dfs.append(df_defect_temp)
    print(f"    Loaded: {csv_file.name} ({len(df_defect_temp)} atoms)")

if not defect_dfs:
    print("\n    ERROR: No defect classification CSVs found!")
    print(f"    Please check the path: {defect_root}")
    exit(1)

df_defect = pd.concat(defect_dfs, ignore_index=True)
print(f"\n    Total atoms in defect classification: {len(df_defect)}")
print(f"    Defect class distribution:")
for defect_class, count in df_defect['Defect_Class'].value_counts().sort_index().items():
    defect_name = DEFECT_MAP.get(defect_class, f'Unknown ({defect_class})')
    print(f"      {defect_name}: {count} ({count / len(df_defect) * 100:.1f}%)")

# ----------------------------
# Merge defect and element data
# ----------------------------
print(f"\n[3] Merging defect and element classifications...")

df_merged = pd.merge(
    df_defect,
    df_element[['Identifier', 'Element', 'Element_Number']],
    on='Identifier',
    how='inner'
)

print(f"    Merged data: {len(df_merged)} atoms matched")
print(f"    Match rate: {len(df_merged) / len(df_defect) * 100:.1f}% of defect data")
print(f"                {len(df_merged) / len(df_element) * 100:.1f}% of element data")

# Add defect class names
df_merged['Defect_Type'] = df_merged['Defect_Class'].map(DEFECT_MAP)

# Save merged data
merged_csv = output_dir / "merged_defect_element_data.csv"
df_merged.to_csv(merged_csv, index=False)
print(f"\n    Saved merged data to: {merged_csv}")

# ----------------------------
# Create summary statistics
# ----------------------------
print(f"\n[4] Generating summary statistics...")

# Cross-tabulation
crosstab = pd.crosstab(
    df_merged['Defect_Type'],
    df_merged['Element'],
    normalize='index'
) * 100  # Convert to percentage

print("\n    Element percentages by defect type:")
print(crosstab.round(1))

# Save crosstab
crosstab_csv = output_dir / "defect_element_crosstab.csv"
crosstab.to_csv(crosstab_csv)
print(f"\n    Saved crosstab to: {crosstab_csv}")

# Count data (for labeling)
count_tab = pd.crosstab(df_merged['Defect_Type'], df_merged['Element'])

# ----------------------------
# Create stacked bar chart
# ----------------------------
print(f"\n[5] Creating stacked histogram...")

# Ensure defect types are in the correct order
defect_order = [
    'No Defect',
    'Interstitial',
    '1 Adjacent Vacancy',
    '2 Adjacent Vacancies',
    '3 Adjacent Vacancies',
    'Others'
]

# Filter to only include defect types present in data
defect_order = [d for d in defect_order if d in crosstab.index]

# Reorder the crosstab
crosstab_ordered = crosstab.loc[defect_order]

# Get element order (sort by overall abundance)
element_counts = df_merged['Element'].value_counts()
element_order = [e for e in element_counts.index if e in crosstab_ordered.columns]

# Reorder columns
crosstab_ordered = crosstab_ordered[element_order]

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(12, 7))

# Plot stacked bars
crosstab_ordered.plot(
    kind='bar',
    stacked=True,
    ax=ax,
    color=[ELEMENT_COLORS.get(e, '#888888') for e in element_order],
    width=0.7,
    edgecolor='black',
    linewidth=0.5
)

# Customize plot
ax.set_xlabel('Defect Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Element Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Element Distribution by Defect Type', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.set_xticklabels(defect_order, rotation=45, ha='right')
ax.legend(title='Element', fontsize=10, title_fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add count annotations below x-axis labels
for i, defect_type in enumerate(defect_order):
    total_count = count_tab.loc[defect_type].sum()
    ax.text(i, -8, f'n={int(total_count)}',
            ha='center', va='top', fontsize=9, style='italic')

plt.tight_layout()

# Save figure
stacked_bar_path = output_dir / "element_by_defect_stacked_bar.png"
plt.savefig(stacked_bar_path, dpi=300, bbox_inches='tight')
print(f"    Saved stacked bar chart to: {stacked_bar_path}")
plt.close()

# ----------------------------
# Create a second version with actual counts (optional)
# ----------------------------
print(f"\n[6] Creating stacked histogram with counts...")

fig, ax = plt.subplots(figsize=(12, 7))

count_tab_ordered = count_tab.loc[defect_order][element_order]

count_tab_ordered.plot(
    kind='bar',
    stacked=True,
    ax=ax,
    color=[ELEMENT_COLORS.get(e, '#888888') for e in element_order],
    width=0.7,
    edgecolor='black',
    linewidth=0.5
)

ax.set_xlabel('Defect Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Atom Count', fontsize=12, fontweight='bold')
ax.set_title('Element Distribution by Defect Type (Counts)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticklabels(defect_order, rotation=45, ha='right')
ax.legend(title='Element', fontsize=10, title_fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

stacked_bar_counts_path = output_dir / "element_by_defect_stacked_bar_counts.png"
plt.savefig(stacked_bar_counts_path, dpi=300, bbox_inches='tight')
print(f"    Saved stacked bar chart (counts) to: {stacked_bar_counts_path}")
plt.close()

# ----------------------------
# Create individual defect-element breakdowns
# ----------------------------
print(f"\n[7] Creating individual defect type breakdowns...")

breakdown_dir = output_dir / "defect_breakdowns"
breakdown_dir.mkdir(parents=True, exist_ok=True)

for defect_type in defect_order:
    subset = df_merged[df_merged['Defect_Type'] == defect_type]
    element_counts = subset['Element'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [ELEMENT_COLORS.get(e, '#888888') for e in element_counts.index]
    bars = ax.bar(element_counts.index, element_counts.values, color=colors,
                  edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Element', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{defect_type}\n(Total: {len(subset)} atoms)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    safe_name = defect_type.replace(' ', '_').replace('/', '_')
    breakdown_path = breakdown_dir / f"{safe_name}_element_breakdown.png"
    plt.savefig(breakdown_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {breakdown_path.name}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print(f"  1. {merged_csv.name} - Combined data")
print(f"  2. {crosstab_csv.name} - Element % by defect type")
print(f"  3. {stacked_bar_path.name} - Main stacked bar chart")
print(f"  4. {stacked_bar_counts_path.name} - Stacked bar chart (counts)")
print(f"  5. defect_breakdowns/ - Individual defect type charts")
print("\n" + "=" * 60)