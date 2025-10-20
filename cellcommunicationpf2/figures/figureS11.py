"""
Figure S11: Alluvial diagram showing flow from ALAD status to CLAD outcomes.
"""

from .common import (
    subplotLabel,
    getSetup,
)
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def makeFigure():
    ax, f = getSetup((14, 10), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    
    # Get unique samples with their ALAD status and outcome information
    patient_df = X.obs[["dsco_id", "ALADstatus", "1yearcondition"]].drop_duplicates(subset=["dsco_id"]).reset_index(drop=True)
    
    print(f"Total unique samples before filtering: {len(patient_df)}")
    print(f"Samples with missing ALADstatus: {patient_df['ALADstatus'].isna().sum()}")
    print(f"Samples with missing 1yearcondition: {patient_df['1yearcondition'].isna().sum()}")
    
    # Instead of dropping all NaN, handle them more carefully
    # Keep samples with ALADstatus but handle missing outcomes
    patient_df = patient_df.dropna(subset=['ALADstatus'])  # Only drop if ALAD status is missing
    
    # Handle categorical column - convert to string first, then fill NaN
    if pd.api.types.is_categorical_dtype(patient_df['1yearcondition']):
        patient_df['1yearcondition'] = patient_df['1yearcondition'].astype(str)
    
    # Fill missing outcomes with 'Unknown' instead of dropping
    patient_df['1yearcondition'] = patient_df['1yearcondition'].fillna('Unknown')
    
    print(f"Total samples after filtering: {len(patient_df)}")
    
    # Simplify ALAD status grouping
    def categorize_alad_status(status):
        status_str = str(status).lower()
        if 'recovered' in status_str:
            return 'Recovered'
        elif 'declined' in status_str:
            return 'Declined'
        else:
            return 'Control'
    
    # Simplify outcome grouping
    def categorize_outcome(condition):
        condition_str = str(condition).lower()
        if condition_str == 'nan' or condition_str == 'unknown':
            return 'Unknown'
        elif 'died' in condition_str:
            return 'Died'
        elif 'clad' in condition_str and ('clad/stable' in condition_str or 'clad/declined' in condition_str):
            return 'CLAD'
        elif 'no clad' in condition_str or 'recovered' in condition_str:
            return 'No CLAD'
        else:
            return 'Other'
    
    # Apply categorization
    patient_df['ALAD_Status'] = patient_df['ALADstatus'].apply(categorize_alad_status)
    print(np.unique(patient_df['1yearcondition']))
    patient_df['Outcome'] = patient_df['1yearcondition'].apply(categorize_outcome)
    
    print(f"\nBefore removing 'Other' categories: {len(patient_df)} samples")
    print("ALAD Status distribution (before filtering):")
    print(patient_df['ALAD_Status'].value_counts())
    print("Outcome distribution (before filtering):")
    print(patient_df['Outcome'].value_counts())
    
    # Only remove 'Other' from ALAD status, keep all outcomes including 'Unknown' and 'Other'
    patient_df = patient_df[patient_df['ALAD_Status'] != 'Other']
    
    print(f"\nAfter removing 'Other' ALAD status: {len(patient_df)} samples")
    
    # Show what we're actually working with
    print("Raw ALADstatus values:")
    print(X.obs['ALADstatus'].value_counts(dropna=False))
    print("\nRaw 1yearcondition values:")
    print(X.obs['1yearcondition'].value_counts(dropna=False))
    
    # Print summary of the simplified data
    print("Simplified ALAD Status distribution:")
    print(patient_df["ALAD_Status"].value_counts())
    print("\nSimplified Outcome distribution:")
    print(patient_df["Outcome"].value_counts())
    
    # Create cross-tabulation
    crosstab = pd.crosstab(patient_df["ALAD_Status"], patient_df["Outcome"])
    print("\nCross-tabulation (ALAD Status vs Outcome):")
    print(crosstab)
    
    # Create patient flow diagram
    create_patient_flow_diagram(ax[0], patient_df, crosstab)
    

    
    return f


def create_patient_flow_diagram(ax, patient_df, crosstab):
    """Create a flow diagram showing patient movement from ALAD status to outcomes."""
    
    # Get unique categories
    alad_statuses = sorted(patient_df['ALAD_Status'].unique())
    outcomes = sorted(patient_df['Outcome'].unique())
    
    # Calculate positions
    total_patients = len(patient_df)
    alad_counts = patient_df['ALAD_Status'].value_counts()
    outcome_counts = patient_df['Outcome'].value_counts()
    
    # Color schemes
    alad_colors = {'Recovered': '#2E8B57', 'Declined': '#DC143C', 'Control': '#4682B4'}  # Green, Red, Blue
    outcome_colors = {'No CLAD': '#90EE90', 'CLAD': '#FFB6C1', 'Died': '#FF6347', 'Unknown': '#D3D3D3', 'Other': '#FFFFE0'}
    
    # Left side - ALAD Status boxes
    left_x = 0.15
    box_width = 0.12
    total_height = 0.7
    
    # Calculate box heights proportional to patient counts
    alad_y_positions = {}
    current_y = 0.85
    
    for status in alad_statuses:
        count = alad_counts[status]
        box_height = (count / total_patients) * total_height
        
        # Draw ALAD status box
        rect = plt.Rectangle((left_x, current_y - box_height), box_width, box_height,
                           facecolor=alad_colors.get(status, '#gray'), 
                           edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        
        # Add label and count
        ax.text(left_x - 0.02, current_y - box_height/2, f'{status}\n(n={count})', 
               ha='right', va='center', fontsize=11, fontweight='bold')
        
        alad_y_positions[status] = current_y - box_height/2
        current_y -= box_height + 0.03
    
    # Right side - Outcome boxes
    right_x = 0.73
    outcome_y_positions = {}
    current_y = 0.85
    
    for outcome in outcomes:
        count = outcome_counts[outcome]
        box_height = (count / total_patients) * total_height
        
        # Draw outcome box
        rect = plt.Rectangle((right_x, current_y - box_height), box_width, box_height,
                           facecolor=outcome_colors.get(outcome, '#gray'),
                           edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        
        # Add label and count
        ax.text(right_x + box_width + 0.02, current_y - box_height/2, f'{outcome}\n(n={count})', 
               ha='left', va='center', fontsize=11, fontweight='bold')
        
        outcome_y_positions[outcome] = current_y - box_height/2
        current_y -= box_height + 0.03
    
    # Draw flow connections
    for alad_status in alad_statuses:
        for outcome in outcomes:
            flow_count = crosstab.loc[alad_status, outcome] if outcome in crosstab.columns else 0
            
            if flow_count > 0:
                # Calculate flow thickness
                flow_thickness = (flow_count / total_patients) * 0.3
                
                # Start and end points
                start_x = left_x + box_width
                start_y = alad_y_positions[alad_status]
                end_x = right_x
                end_y = outcome_y_positions[outcome]
                
                # Create curved connection using bezier curve
                mid_x = (start_x + end_x) / 2
                
                # Draw flow band using polygon
                flow_top_start = start_y + flow_thickness/2
                flow_bottom_start = start_y - flow_thickness/2
                flow_top_end = end_y + flow_thickness/2
                flow_bottom_end = end_y - flow_thickness/2
                
                # Create smooth curve points
                x_curve = np.linspace(start_x, end_x, 100)
                y_top_curve = np.interp(x_curve, [start_x, mid_x, end_x], 
                                       [flow_top_start, (flow_top_start + flow_top_end)/2, flow_top_end])
                y_bottom_curve = np.interp(x_curve, [start_x, mid_x, end_x],
                                          [flow_bottom_start, (flow_bottom_start + flow_bottom_end)/2, flow_bottom_end])
                
                # Create polygon for flow band
                x_poly = np.concatenate([x_curve, x_curve[::-1]])
                y_poly = np.concatenate([y_top_curve, y_bottom_curve[::-1]])
                
                ax.fill(x_poly, y_poly, color=alad_colors.get(alad_status, '#gray'), 
                       alpha=0.4, edgecolor='none')
                
                # Add flow count label
                if flow_count > 1:  # Only label significant flows
                    label_x = mid_x
                    label_y = (start_y + end_y) / 2
                    ax.text(label_x, label_y, str(flow_count), 
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title and labels
    ax.text(0.5, 0.95, 'Patient Flow: ALAD Status → 1-Year Outcome', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.text(left_x + box_width/2, 0.05, 'ALAD Status', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(right_x + box_width/2, 0.05, '1-Year Outcome', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add total patient count
    ax.text(0.5, 0.02, f'Total Patients: {total_patients}', 
           ha='center', va='center', fontsize=12, style='italic')


