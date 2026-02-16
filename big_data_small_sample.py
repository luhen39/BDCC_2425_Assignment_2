"""
ICU Length of Stay Prediction - Big Data Assignment
Group W - Big Data and Cloud Computing - FCUP - 24/25

Authors:
- Fatima Khalid - 202301625
- Jorge Pereira - 201705314
- Luis Ganco - 202004196

Description:
This script processes clinical time series data from ICU patient records
to build machine learning models for predicting length of stay.
Uses Dask for big data processing of CHARTEVENTS dataset.
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

import pandas as pd
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import gc
import warnings
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/mnt/disks/ssd/'
SAMPLE_SIZE = 1000  # Adjust as needed

print("=" * 80)
print("ICU LENGTH OF STAY PREDICTION - BIG DATA PROCESSING")
print("=" * 80)
print(f"Sample size: {SAMPLE_SIZE} patients")
print(f"Data path: {DATA_PATH}")

# ============================================================================
# STEP 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: DATA LOADING AND INITIAL EXPLORATION")
print("=" * 80)

def load_metadata_files():
    """Load and prepare metadata files"""
    print("\nLoading metadata files...")
    
    # Load small metadata files
    d_items = pd.read_csv(f'{DATA_PATH}D_ITEMS.csv')
    icustays = pd.read_csv(f'{DATA_PATH}ICUSTAYS.csv')
    admissions = pd.read_csv(f'{DATA_PATH}ADMISSIONS.csv')
    
    print(f"D_ITEMS: {len(d_items):,} clinical parameters")
    print(f"ICUSTAYS: {len(icustays):,} ICU stays")
    print(f"ADMISSIONS: {len(admissions):,} hospital admissions")
    
    return d_items, icustays, admissions

def prepare_icu_cohort(icustays, admissions):
    """Prepare ICU stays with length of stay calculation"""
    print("\nPreparing ICU cohort...")
    
    # Filter only ICU stays with non-null times
    icustays = icustays.dropna(subset=['INTIME', 'OUTTIME'])
    
    # Add LOS in hours
    icustays['LOS_HOURS'] = (
        pd.to_datetime(icustays['OUTTIME']) - pd.to_datetime(icustays['INTIME'])
    ).dt.total_seconds() / 3600
    
    # Join with ADMISSIONS for demographics
    icustay_adm = pd.merge(icustays, admissions, on='HADM_ID')
    
    print(f"ICU stays with valid times: {len(icustay_adm):,}")
    print(f"LOS range: {icustay_adm['LOS_HOURS'].min():.1f} - {icustay_adm['LOS_HOURS'].max():.1f} hours")
    print(f"Mean LOS: {icustay_adm['LOS_HOURS'].mean():.1f} hours ({icustay_adm['LOS_HOURS'].mean()/24:.1f} days)")
    
    return icustay_adm

def create_patient_sample(icustay_adm, sample_size):
    """Create random sample of patients"""
    print(f"\nCreating sample of {sample_size} patients...")
    
    sample_ids = icustay_adm.sample(n=sample_size, random_state=42)['ICUSTAY_ID'].tolist()
    
    print(f"Sample created: {len(sample_ids)} patients")
    print(f"Sample percentage: {(len(sample_ids)/len(icustay_adm)*100):.2f}%")
    
    return sample_ids

# Execute Step 1
d_items, icustays, admissions = load_metadata_files()
icustay_adm = prepare_icu_cohort(icustays, admissions)
sample_ids = create_patient_sample(icustay_adm, SAMPLE_SIZE)

# Display sample of data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print("\nSample of ICU stays data:")
print(icustay_adm.head())

# ============================================================================
# STEP 2: BIG DATA PROCESSING WITH DASK
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: BIG DATA PROCESSING WITH DASK")
print("=" * 80)

def load_chartevents_with_dask():
    """Load CHARTEVENTS using Dask for big data processing"""
    print("\nLoading CHARTEVENTS with Dask...")
    
    # Specify dtypes to avoid conversion errors
    dtype_spec = {
        'ERROR': 'float64',
        'ICUSTAY_ID': 'float64',
        'RESULTSTATUS': 'object',
        'STOPPED': 'object',
        'VALUE': 'object',
        'WARNING': 'float64',
        'VALUENUM': 'float64',
        'CHARTTIME': 'object',
        'ITEMID': 'float64',
        'CGID': 'float64'
    }
    
    chartevents_dask = dd.read_csv(
        f"{DATA_PATH}CHARTEVENTS.csv",
        dtype=dtype_spec
    )
    
    print("CHARTEVENTS loaded with Dask (lazy evaluation)")
    return chartevents_dask

def filter_and_process_events(chartevents_dask, sample_ids, icustays):
    """Filter events for sample patients and add temporal features"""
    print("\nFiltering for sampled ICU stays...")
    
    # Filter for sampled ICU stays (lazy operation)
    chartevents_sample = chartevents_dask[chartevents_dask.ICUSTAY_ID.isin(sample_ids)]
    
    # Convert datetime columns
    chartevents_sample['CHARTTIME'] = dd.to_datetime(chartevents_sample['CHARTTIME'], errors='coerce')
    
    print("Dask DataFrame created successfully")
    
    # Convert ICU stays to Dask DataFrame
    icustays_dask = dd.from_pandas(icustays, npartitions=1)
    icustays_dask['INTIME'] = dd.to_datetime(icustays_dask['INTIME'])
    
    print("Joining datasets...")
    # Merge the datasets
    joined = chartevents_sample.merge(
        icustays_dask[['ICUSTAY_ID', 'INTIME']],
        on='ICUSTAY_ID',
        how='inner'
    )
    
    # Calculate hours since admission
    joined['hours_since_admit'] = (
        joined['CHARTTIME'] - joined['INTIME']
    ).dt.total_seconds() / 3600
    
    # Filter for first 12 hours (lazy operation)
    filtered_events_dask = joined[
        (joined['hours_since_admit'] <= 12) &
        (joined['hours_since_admit'] >= 0)
    ]
    
    print("Filtered dataset created (lazy evaluation)")
    return filtered_events_dask

# Execute Step 2
chartevents_dask = load_chartevents_with_dask()
filtered_events_dask = filter_and_process_events(chartevents_dask, sample_ids, icustays)

# ============================================================================
# STEP 3: DATA VISUALIZATION AND EXPLORATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: DATA VISUALIZATION AND EXPLORATION")
print("=" * 80)

def visualize_patient_data(filtered_events_dask, d_items, patient_id=265846):
    """Visualize clinical data for a specific patient"""
    print(f"\nGetting data for patient {patient_id}...")
    
    # Filter for specific patient
    patient_filtered = filtered_events_dask[filtered_events_dask.ICUSTAY_ID == patient_id]
    patient_data = patient_filtered[['CHARTTIME', 'INTIME', 'VALUENUM', 'ITEMID', 'hours_since_admit']]
    subset = patient_data.head(200)
    
    print(f"Found {len(subset)} records for patient {patient_id}")
    
    if len(subset) > 0:
        # Calculate time in days for plotting
        subset['Time'] = subset['hours_since_admit'] / 24
        subset = subset.dropna(subset=['VALUENUM', 'Time'])
        
        def get_item_name(item_id):
            """Get clinical item name from D_ITEMS"""
            item_info = d_items[d_items['ITEMID'] == item_id]
            if not item_info.empty:
                return item_info['LABEL'].iloc[0]
            else:
                return f"Item {int(item_id)}"
        
        # Create plot with clinical names
        plt.figure(figsize=(15, 8))
        
        # Get unique item IDs and limit to top 5
        top_items = subset['ITEMID'].value_counts().head(5).index
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, item_id in enumerate(top_items):
            item_data = subset[subset['ITEMID'] == item_id]
            if len(item_data) > 0:
                item_name = get_item_name(item_id)
                plt.scatter(item_data['Time'], item_data['VALUENUM'],
                           label=f'{item_name} ({int(item_id)})',
                           alpha=0.7, s=30, color=colors[i])
        
        plt.xlabel('Time (days since ICU admission)')
        plt.ylabel('Clinical Values')
        plt.title(f'ICU Patient {int(patient_id)} - First 12 Hours Clinical Data')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Successfully plotted {len(subset)} data points")
        
        # Print detailed item information
        print("\nClinical Items in this plot:")
        print("=" * 40)
        for item_id in top_items:
            item_info = d_items[d_items['ITEMID'] == item_id]
            if not item_info.empty:
                label = item_info['LABEL'].iloc[0]
                category = item_info['CATEGORY'].iloc[0] if 'CATEGORY' in item_info.columns else 'N/A'
                print(f"• {label} (ID: {int(item_id)})")
                print(f"  Category: {category}")
            else:
                print(f"• Item {int(item_id)}: Description not found")

def analyze_dataset_size():
    """Analyze current dataset size and coverage"""
    print("\n=== CURRENT DATASET SIZE ANALYSIS ===")
    
    print("Current sample configuration:")
    print(f"Sample size set: {len(sample_ids):,} ICU stays")
    
    print(f"\nOriginal dataset sizes:")
    print(f"  ICU stays (icustay_adm): {len(icustay_adm):,} patients")
    print(f"  Sample IDs selected: {len(sample_ids):,} patients")
    print(f"  Sample percentage: {(len(sample_ids)/len(icustay_adm)*100):.2f}%")
    
    print(f"\nActual data being processed:")
    print("  CHARTEVENTS: Filtered for sample_ids using Dask")
    print("  Time window: First 12 hours only")
    
    # Calculate estimated data size
    estimated_records_per_patient = 52
    estimated_total_records = len(sample_ids) * estimated_records_per_patient
    
    print(f"\nEstimated processing size:")
    print(f"  Estimated CHARTEVENTS records: ~{estimated_total_records:,}")
    print(f"  (Based on {estimated_records_per_patient} records per patient average)")

# Execute Step 3
visualize_patient_data(filtered_events_dask, d_items)
analyze_dataset_size()

# ============================================================================
# STEP 4: PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: PREPROCESSING AND FEATURE ENGINEERING")
print("=" * 80)

def assess_data_quality(filtered_events_dask):
    """Assess data quality and basic statistics"""
    print("\nData Quality Assessment")
    print("-" * 40)
    
    # Sample data for quality check
    sample_for_quality = filtered_events_dask.head(10000)
    print(f"Quality check sample size: {len(sample_for_quality)}")
    
    # Basic statistics
    print("\nBasic data overview:")
    print(f"  Columns: {sample_for_quality.columns.tolist()}")
    print(f"  Missing VALUENUM: {sample_for_quality['VALUENUM'].isnull().sum()}")
    print(f"  Missing ITEMID: {sample_for_quality['ITEMID'].isnull().sum()}")
    print(f"  Unique ITEMIDs in sample: {sample_for_quality['ITEMID'].nunique()}")
    
    # Check value ranges
    valid_numeric = sample_for_quality[sample_for_quality['VALUENUM'].notna()]
    if len(valid_numeric) > 0:
        print(f"  VALUENUM range: [{valid_numeric['VALUENUM'].min():.2f}, {valid_numeric['VALUENUM'].max():.2f}]")
        print(f"  VALUENUM mean: {valid_numeric['VALUENUM'].mean():.2f}")

def get_top_clinical_items(events_df, d_items, n_items=15):
    """Get the most frequently measured clinical parameters"""
    print("\nFeature Engineering Strategy")
    print("-" * 40)
    
    # Sample data to get representative item frequencies
    sample_data = events_df.head(50000)
    
    # Count occurrences of each ITEMID
    item_counts = sample_data['ITEMID'].value_counts().head(n_items)
    
    print(f"Top {n_items} most frequent clinical measurements:")
    for i, (item_id, count) in enumerate(item_counts.items(), 1):
        # Get item name from D_ITEMS
        item_info = d_items[d_items['ITEMID'] == item_id]
        item_name = item_info['LABEL'].iloc[0] if not item_info.empty else f"Item {item_id}"
        print(f"  {i:2d}. {item_name} (ID: {int(item_id)}) - {count} measurements")
    
    return item_counts.index.tolist()

def get_patients_with_data(events_df, min_measurements=5):
    """Identify patients who have sufficient measurement data"""
    print("\nIdentifying patients with measurement data...")
    
    # Get larger sample to find all patients with data
    large_sample = events_df.head(20000)
    
    if len(large_sample) > 0:
        # Count measurements per patient
        patient_counts = large_sample['ICUSTAY_ID'].value_counts()
        
        # Filter patients with minimum measurements
        patients_with_sufficient_data = patient_counts[patient_counts >= min_measurements].index.tolist()
        
        print(f"Found {len(patients_with_sufficient_data)} patients with >= {min_measurements} measurements")
        print(f"Patient measurement counts:")
        for patient_id in patients_with_sufficient_data[:10]:
            count = patient_counts[patient_id]
            print(f"  Patient {int(patient_id)}: {count} measurements")
        
        return patients_with_sufficient_data
    else:
        return []

def create_patient_features_dask(events_df, selected_items):
    """Create patient-level features using data-driven patient selection"""
    print("\nPatient-Level Feature Aggregation")
    print("-" * 40)
    
    print("Creating patient-level features...")
    
    # Get patients who have measurement data
    patients_with_data = get_patients_with_data(events_df, min_measurements=5)
    
    if len(patients_with_data) == 0:
        print("No patients found with sufficient data")
        return pd.DataFrame()
    
    # Filter to patients and items with valid numeric values
    filtered_df = events_df[
        (events_df['ICUSTAY_ID'].isin(patients_with_data)) &
        (events_df['ITEMID'].isin(selected_items)) &
        (events_df['VALUENUM'].notnull())
    ]
    
    patient_features_list = []
    
    # Process all items for confirmed patients
    for item_id in selected_items:
        print(f"  Processing ITEMID {int(item_id)}...")
        
        item_data = filtered_df[filtered_df['ITEMID'] == item_id]
        item_sample = item_data.head(10000)
        
        if len(item_sample) > 0:
            patient_item_stats = item_sample.groupby('ICUSTAY_ID')['VALUENUM'].agg([
                'count', 'mean', 'std', 'min', 'max', 'first', 'last'
            ]).reset_index()
            
            if len(patient_item_stats) > 0:
                patient_item_stats['ITEMID'] = item_id
                patient_features_list.append(patient_item_stats)
                print(f"    Created features for {len(patient_item_stats)} patients")
    
    # Combine all features
    if patient_features_list:
        all_features = pd.concat(patient_features_list, ignore_index=True)
        print(f"\nCombined features from {len(patient_features_list)} measurement types")
        return all_features
    else:
        print("No features created across any measurement types")
        return pd.DataFrame()

# Execute Step 4 - Part 1
assess_data_quality(filtered_events_dask)
top_items = get_top_clinical_items(filtered_events_dask, d_items, n_items=15)
patient_features = create_patient_features_dask(filtered_events_dask, top_items)

# Report feature creation results
if len(patient_features) > 0:
    print(f"\nFeature matrix info:")
    print(f"  Shape: {patient_features.shape}")
    print(f"  Unique patients: {patient_features['ICUSTAY_ID'].nunique()}")
    print(f"  Items covered: {patient_features['ITEMID'].nunique()}")
    
    # Show patients included
    unique_patients = patient_features['ICUSTAY_ID'].unique()
    print(f"\nPatients included in feature matrix:")
    for i, patient_id in enumerate(sorted(unique_patients), 1):
        patient_measurements = len(patient_features[patient_features['ICUSTAY_ID'] == patient_id])
        print(f"  {i:2d}. Patient {int(patient_id)}: {patient_measurements} feature combinations")
    
    # Show feature coverage analysis
    coverage_stats = patient_features.groupby('ICUSTAY_ID').size()
    print(f"\nFeature coverage analysis:")
    print(f"  Mean features per patient: {coverage_stats.mean():.1f}")
    print(f"  Min features per patient: {coverage_stats.min()}")
    print(f"  Max features per patient: {coverage_stats.max()}")
else:
    print("No features were created - debugging needed")

# ============================================================================
# STEP 5: FEATURE MATRIX CREATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: FEATURE MATRIX CREATION")
print("=" * 80)

def pivot_features_for_ml(features_df, d_items):
    """Create wide-format feature matrix suitable for machine learning"""
    print("\nCreating ML-Ready Feature Matrix")
    print("-" * 40)
    
    print("Pivoting features to wide format...")
    
    # Create pivot table for mean values
    pivot_mean = features_df.pivot_table(
        index='ICUSTAY_ID',
        columns='ITEMID',
        values='mean',
        fill_value=np.nan
    )
    
    # Rename columns with descriptive names
    renamed_cols = {}
    for item_id in pivot_mean.columns:
        item_info = d_items[d_items['ITEMID'] == item_id]
        if not item_info.empty:
            item_name = item_info['LABEL'].iloc[0].replace(' ', '_').replace('/', '_')
            renamed_cols[item_id] = f"mean_{item_name}_{int(item_id)}"
        else:
            renamed_cols[item_id] = f"mean_item_{int(item_id)}"
    
    pivot_mean = pivot_mean.rename(columns=renamed_cols)
    
    # Add measurement counts as features
    pivot_count = features_df.pivot_table(
        index='ICUSTAY_ID',
        columns='ITEMID',
        values='count',
        fill_value=0
    )
    
    # Rename count columns
    count_cols = {}
    for item_id in pivot_count.columns:
        item_info = d_items[d_items['ITEMID'] == item_id]
        if not item_info.empty:
            item_name = item_info['LABEL'].iloc[0].replace(' ', '_').replace('/', '_')
            count_cols[item_id] = f"count_{item_name}_{int(item_id)}"
        else:
            count_cols[item_id] = f"count_item_{int(item_id)}"
    
    pivot_count = pivot_count.rename(columns=count_cols)
    
    # Combine mean and count features
    ml_features = pd.concat([pivot_mean, pivot_count], axis=1)
    
    return ml_features

# Create ML-ready features
if len(patient_features) > 0:
    ml_features = pivot_features_for_ml(patient_features, d_items)
    
    print(f"ML feature matrix created:")
    print(f"  Shape: {ml_features.shape}")
    print(f"  Features: {list(ml_features.columns)}")
    
    print("\nSample feature matrix:")
    print(ml_features.head())
else:
    print("Cannot create ML features - no patient features available")
    ml_features = pd.DataFrame()

# ============================================================================
# STEP 6: DATA QUALITY AND PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: DATA QUALITY AND PREPROCESSING")
print("=" * 80)

def detect_outliers_and_validate(ml_features):
    """Enhanced outlier detection with clinical validation"""
    print("=== ENHANCED DATA VALIDATION & OUTLIER DETECTION ===")
    
    # Define clinical normal ranges
    clinical_ranges = {
        'Heart_Rate': (60, 100),
        'Systolic_BP': (90, 140),
        'Diastolic_BP': (60, 90),
        'Respiratory_Rate': (12, 20),
        'SpO2': (95, 100),
        'Temperature': (97, 99),
        'Glucose': (70, 140)
    }
    
    outliers_summary = {}
    clinical_violations = {}
    
    print("Processing all features for outlier detection...")
    
    for feature in ml_features.columns:
        if 'mean_' in feature:
            values = ml_features[feature].dropna()
            if len(values) > 0:
                # Statistical outliers (IQR method)
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    outlier_mask = (values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR))
                    outlier_patients = values[outlier_mask].index.tolist()
                else:
                    outlier_mask = pd.Series([False] * len(values), index=values.index)
                    outlier_patients = []
                
                outliers_summary[feature] = {
                    'statistical_outliers': outlier_mask.sum(),
                    'outlier_patients': outlier_patients,
                    'outlier_values': values[outlier_mask].tolist() if outlier_mask.sum() > 0 else [],
                    'min': values.min(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'std': values.std(),
                    'Q1': Q1,
                    'Q3': Q3
                }
                
                # Clinical range validation
                clinical_violations[feature] = check_clinical_ranges(feature, values, clinical_ranges)
    
    # Print results
    print_outlier_analysis(outliers_summary, clinical_violations)
    
    return outliers_summary, clinical_violations

def check_clinical_ranges(feature_name, values, clinical_ranges):
    """Check if values fall within expected clinical ranges"""
    violations = {'low': [], 'high': [], 'patients_low': [], 'patients_high': []}
    
    # Map feature names to clinical parameters
    range_key = None
    if 'Heart_Rate' in feature_name:
        range_key = 'Heart_Rate'
    elif 'systolic' in feature_name.lower():
        range_key = 'Systolic_BP'
    elif 'diastolic' in feature_name.lower():
        range_key = 'Diastolic_BP'
    elif 'Respiratory' in feature_name:
        range_key = 'Respiratory_Rate'
    elif 'saturation' in feature_name.lower() or 'SpO2' in feature_name:
        range_key = 'SpO2'
    elif 'Temperature' in feature_name:
        range_key = 'Temperature'
    elif 'Glucose' in feature_name:
        range_key = 'Glucose'
    
    if range_key and range_key in clinical_ranges:
        min_normal, max_normal = clinical_ranges[range_key]
        
        # Find values outside normal range
        low_mask = values < min_normal
        high_mask = values > max_normal
        
        violations['low'] = values[low_mask].tolist()
        violations['high'] = values[high_mask].tolist()
        violations['patients_low'] = values[low_mask].index.tolist()
        violations['patients_high'] = values[high_mask].index.tolist()
    
    return violations

def print_outlier_analysis(outliers_summary, clinical_violations):
    """Print detailed outlier analysis results"""
    print("\n" + "=" * 60)
    print("STATISTICAL OUTLIER ANALYSIS (IQR Method)")
    print("=" * 60)
    
    total_outliers = 0
    for feature, info in outliers_summary.items():
        if info['statistical_outliers'] > 0:
            total_outliers += info['statistical_outliers']
            print(f"\n[OUTLIER] {feature}:")
            print(f"   Outliers: {info['statistical_outliers']}")
            print(f"   Range: {info['min']:.1f} - {info['max']:.1f}")
            print(f"   Mean +/- SD: {info['mean']:.1f} +/- {info['std']:.1f}")
            print(f"   Normal range (Q1-Q3): {info['Q1']:.1f} - {info['Q3']:.1f}")
            
            if info['outlier_patients']:
                print(f"   Outlier patients: {[int(p) for p in info['outlier_patients']]}")
                print(f"   Outlier values: {[f'{v:.1f}' for v in info['outlier_values']]}")
    
    print(f"\nSUMMARY: {total_outliers} statistical outliers found")
    
    print("\n" + "=" * 60)
    print("CLINICAL RANGE VALIDATION")
    print("=" * 60)
    
    total_clinical_violations = 0
    for feature, violations in clinical_violations.items():
        total_violations = len(violations['low']) + len(violations['high'])
        if total_violations > 0:
            total_clinical_violations += total_violations
            print(f"\n[CLINICAL] {feature}:")
            
            if violations['low']:
                print(f"   Below normal: {len(violations['low'])} values")
                print(f"   Patients: {[int(p) for p in violations['patients_low']]}")
                print(f"   Values: {[f'{v:.1f}' for v in violations['low']]}")
            
            if violations['high']:
                print(f"   Above normal: {len(violations['high'])} values")
                print(f"   Patients: {[int(p) for p in violations['patients_high']]}")
                print(f"   Values: {[f'{v:.1f}' for v in violations['high']]}")
    
    print(f"\nSUMMARY: {total_clinical_violations} clinical range violations found")
    
    if total_outliers == 0 and total_clinical_violations == 0:
        print("\nNo significant outliers or clinical violations detected!")
        print("Data appears to be within expected ranges for ICU patients.")

def handle_missing_values(ml_features):
    """Comprehensive missing value analysis and imputation"""
    print("\n=== MISSING VALUE ANALYSIS ===")
    
    # Calculate missing value percentages
    missing_percent = (ml_features.isnull().sum() / len(ml_features) * 100).sort_values(ascending=False)
    
    print("Missing value percentages by feature:")
    print("-" * 50)
    for feature, percent in missing_percent.items():
        if percent > 0:
            print(f"  {feature}: {percent:.1f}%")
    
    if missing_percent.max() == 0:
        print("  No missing values detected in any features!")
        return ml_features.copy()
    
    # Create imputation strategy
    ml_features_imputed = ml_features.copy()
    
    # Strategy 1: Zero imputation for counts (missing = not measured)
    count_features = [col for col in ml_features.columns if 'count_' in col]
    if count_features:
        before_count = ml_features_imputed[count_features].isnull().sum().sum()
        ml_features_imputed[count_features] = ml_features_imputed[count_features].fillna(0)
        print(f"Count features: Filled {before_count} missing values with 0")
    
    # Strategy 2: Mean imputation for mean values
    mean_features = [col for col in ml_features.columns if 'mean_' in col]
    if mean_features:
        for col in mean_features:
            if ml_features_imputed[col].isnull().sum() > 0:
                mean_val = ml_features_imputed[col].mean()
                ml_features_imputed[col] = ml_features_imputed[col].fillna(mean_val)
                print(f"  {col}: Filled with mean value {mean_val:.1f}")
    
    # Final check
    total_missing_after = ml_features_imputed.isnull().sum().sum()
    print(f"\nIMPUTATION SUMMARY:")
    print(f"  Missing values before: {ml_features.isnull().sum().sum()}")
    print(f"  Missing values after: {total_missing_after}")
    print(f"  Features processed: {len(ml_features.columns)}")
    
    if total_missing_after == 0:
        print("  All missing values successfully imputed!")
    else:
        print(f"  {total_missing_after} missing values remain")
    
    return ml_features_imputed

def scale_features(features_df):
    """Scale features for machine learning"""
    print("\n=== FEATURE SCALING ===")
    
    # Separate count and mean features for different scaling
    count_features = [col for col in features_df.columns if 'count_' in col]
    mean_features = [col for col in features_df.columns if 'mean_' in col]
    
    scaled_features = features_df.copy()
    
    # StandardScaler for mean values
    if mean_features:
        scaler_mean = StandardScaler()
        scaled_features[mean_features] = scaler_mean.fit_transform(features_df[mean_features])
        print(f"Standard scaled {len(mean_features)} mean features")
        print("   Mean features now have mean~0, std~1")
    
    # MinMaxScaler for count features
    if count_features:
        scaler_count = MinMaxScaler()
        scaled_features[count_features] = scaler_count.fit_transform(features_df[count_features])
        print(f"MinMax scaled {len(count_features)} count features")
        print("   Count features now range [0,1]")
    
    print("Feature scaling complete")
    
    return scaled_features, scaler_mean if mean_features else None, scaler_count if count_features else None

# Execute Step 6
if len(ml_features) > 0:
    outliers_info, clinical_info = detect_outliers_and_validate(ml_features)
    ml_features_clean = handle_missing_values(ml_features)
    ml_features_scaled, scaler_mean, scaler_count = scale_features(ml_features_clean)
    
    print("\n" + "=" * 60)
    print("CLEANED AND SCALED FEATURE MATRIX READY")
    print("=" * 60)
    print(f"Shape: {ml_features_scaled.shape}")
    print("\nSample scaled features:")
    print(ml_features_scaled.head())
else:
    print("No features available for processing")

# ============================================================================
# STEP 7: TEMPORAL FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: TEMPORAL FEATURE ENGINEERING")
print("=" * 80)

def create_temporal_features(filtered_events_dask, ml_features_clean):
    """Create advanced temporal features"""
    print("=== TEMPORAL FEATURE ENGINEERING ===")
    
    temporal_features = []
    
    print("Analyzing temporal patterns for each patient...")
    
    for patient_id in ml_features_clean.index:
        print(f"  Processing Patient {int(patient_id)}...")
        
        # Filter patient data from Dask DataFrame
        patient_data = filtered_events_dask[filtered_events_dask.ICUSTAY_ID == patient_id].head(1000)
        
        if len(patient_data) > 0:
            # Calculate temporal statistics
            hours_data = patient_data['hours_since_admit'].dropna()
            
            if len(hours_data) > 0:
                # Time-based features
                features = {
                    'ICUSTAY_ID': patient_id,
                    'first_measurement_hour': hours_data.min(),
                    'last_measurement_hour': hours_data.max(),
                    'measurement_span_hours': hours_data.max() - hours_data.min(),
                    'total_measurements': len(patient_data),
                    'measurements_per_hour': len(patient_data) / max(1, hours_data.max()),
                    'unique_measurement_types': patient_data['ITEMID'].nunique(),
                    'measurement_frequency': len(patient_data) / max(1, len(hours_data.unique())),
                    'monitoring_intensity': patient_data['ITEMID'].nunique() / max(1, hours_data.max())
                }
                temporal_features.append(features)
                
                print(f"    Created features: {len(patient_data)} measurements, {hours_data.max():.1f}h span")
            else:
                print(f"    No valid time data")
        else:
            print(f"    No measurement data")
    
    if temporal_features:
        temporal_df = pd.DataFrame(temporal_features).set_index('ICUSTAY_ID')
        
        # Combine with existing features
        enhanced_features = pd.concat([ml_features_clean, temporal_df], axis=1)
        
        print(f"\nTEMPORAL FEATURES SUMMARY:")
        print(f"  Added {len(temporal_df.columns)} temporal features")
        print(f"  Enhanced matrix shape: {enhanced_features.shape}")
        
        # Show temporal feature statistics
        print(f"\nTemporal feature statistics:")
        for col in temporal_df.columns:
            values = temporal_df[col].dropna()
            if len(values) > 0:
                print(f"  {col}:")
                print(f"    Range: {values.min():.1f} - {values.max():.1f}")
                print(f"    Mean: {values.mean():.1f}")
        
        return enhanced_features
    else:
        print("No temporal features could be created")
        return ml_features_clean

# Execute Step 7
if len(ml_features_clean) > 0:
    ml_features_enhanced = create_temporal_features(filtered_events_dask, ml_features_clean)
    
    print("\n" + "=" * 60)
    print("ENHANCED FEATURE MATRIX WITH TEMPORAL FEATURES")
    print("=" * 60)
    print(f"Final shape: {ml_features_enhanced.shape}")
    print(f"Total features: {ml_features_enhanced.shape[1]}")
    
    # Show sample of enhanced features
    print("\nSample enhanced features (first 5 columns):")
    print(ml_features_enhanced.iloc[:, :5].head())
    
    print("\nTemporal features (last columns):")
    temporal_cols = ml_features_enhanced.columns[-8:] if ml_features_enhanced.shape[1] >= 8 else ml_features_enhanced.columns[-5:]
    print(ml_features_enhanced[temporal_cols].head())
else:
    print("No features available for temporal enhancement")
    ml_features_enhanced = pd.DataFrame()

# ============================================================================
# STEP 8: TARGET VARIABLE PREPARATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: TARGET VARIABLE PREPARATION")
print("=" * 80)

def prepare_target_variable(ml_features_enhanced, icustay_adm):
    """Prepare the target variable (LOS_HOURS) for prediction"""
    print("=== TARGET VARIABLE PREPARATION ===")
    
    # Get target variable for patients in feature matrix
    target_data = icustay_adm[icustay_adm['ICUSTAY_ID'].isin(ml_features_enhanced.index)][['ICUSTAY_ID', 'LOS_HOURS']]
    target_data = target_data.set_index('ICUSTAY_ID')
    
    print(f"Target variable statistics:")
    print(f"  Patients with LOS data: {len(target_data)}")
    print(f"  LOS range: {target_data['LOS_HOURS'].min():.1f} - {target_data['LOS_HOURS'].max():.1f} hours")
    print(f"  LOS mean: {target_data['LOS_HOURS'].mean():.1f} hours")
    print(f"  LOS median: {target_data['LOS_HOURS'].median():.1f} hours")
    print(f"  LOS standard deviation: {target_data['LOS_HOURS'].std():.1f} hours")
    
    # Create categorical target for classification
    target_data['LOS_CATEGORY'] = pd.cut(
        target_data['LOS_HOURS'],
        bins=[0, 24, 72, 168, float('inf')],
        labels=['Short (<1d)', 'Medium (1-3d)', 'Long (3-7d)', 'Very Long (>7d)']
    )
    
    # Show distribution of categories
    print(f"\nLOS category distribution:")
    category_counts = target_data['LOS_CATEGORY'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(target_data)) * 100
        print(f"  {category}: {count} patients ({percentage:.1f}%)")
    
    # Show individual patient LOS values
    print(f"\nIndividual patient LOS values:")
    for patient_id in target_data.index:
        los_hours = target_data.loc[patient_id, 'LOS_HOURS']
        los_days = los_hours / 24
        category = target_data.loc[patient_id, 'LOS_CATEGORY']
        print(f"  Patient {int(patient_id)}: {los_hours:.1f} hours ({los_days:.1f} days) - {category}")
    
    return target_data

def create_final_dataset(features, targets):
    """Combine features and targets into final ML dataset"""
    print("\n=== FINAL DATASET ASSEMBLY ===")
    
    # Ensure alignment
    common_patients = set(features.index) & set(targets.index)
    print(f"Common patients between features and targets: {len(common_patients)}")
    
    if len(common_patients) == 0:
        print("No common patients found between features and targets!")
        return pd.DataFrame()
    
    # Filter both datasets to common patients
    features_aligned = features.loc[list(common_patients)]
    targets_aligned = targets.loc[list(common_patients)]
    
    # Combine features and targets
    final_dataset = pd.concat([features_aligned, targets_aligned], axis=1)
    
    # Remove missing target values
    before_cleaning = len(final_dataset)
    final_dataset = final_dataset.dropna(subset=['LOS_HOURS'])
    after_cleaning = len(final_dataset)
    
    if before_cleaning > after_cleaning:
        print(f"Removed {before_cleaning - after_cleaning} rows with missing LOS_HOURS")
    
    print(f"\nFINAL DATASET SUMMARY:")
    print(f"  Shape: {final_dataset.shape}")
    print(f"  Patients: {final_dataset.shape[0]}")
    print(f"  Features: {final_dataset.shape[1] - 2} (excluding LOS_HOURS and LOS_CATEGORY)")
    print(f"  Target variable: LOS_HOURS (continuous)")
    print(f"  Target categories: LOS_CATEGORY (classification)")
    
    # Show feature breakdown
    feature_cols = [col for col in final_dataset.columns if col not in ['LOS_HOURS', 'LOS_CATEGORY']]
    mean_features = [col for col in feature_cols if 'mean_' in col]
    count_features = [col for col in feature_cols if 'count_' in col]
    temporal_features = [col for col in feature_cols if col in ['first_measurement_hour', 'last_measurement_hour',
                                                              'measurement_span_hours', 'total_measurements',
                                                              'measurements_per_hour', 'unique_measurement_types',
                                                              'measurement_frequency', 'monitoring_intensity']]
    
    print(f"\nFEATURE BREAKDOWN:")
    print(f"  Clinical mean values: {len(mean_features)} features")
    print(f"  Measurement counts: {len(count_features)} features")
    print(f"  Temporal features: {len(temporal_features)} features")
    print(f"  Total features: {len(feature_cols)} features")
    
    # Show target distribution
    print(f"\nTARGET VARIABLE DISTRIBUTION:")
    print(f"  LOS range: {final_dataset['LOS_HOURS'].min():.1f} - {final_dataset['LOS_HOURS'].max():.1f} hours")
    print(f"  LOS mean: {final_dataset['LOS_HOURS'].mean():.1f} hours ({final_dataset['LOS_HOURS'].mean()/24:.1f} days)")
    print(f"  LOS median: {final_dataset['LOS_HOURS'].median():.1f} hours ({final_dataset['LOS_HOURS'].median()/24:.1f} days)")
    
    category_dist = final_dataset['LOS_CATEGORY'].value_counts()
    print(f"\n  Category distribution:")
    for category, count in category_dist.items():
        percentage = (count / len(final_dataset)) * 100
        print(f"    {category}: {count} patients ({percentage:.1f}%)")
    
    return final_dataset

# Execute Step 8
if len(ml_features_enhanced) > 0:
    target_variables = prepare_target_variable(ml_features_enhanced, icustay_adm)
    
    # Verify alignment
    print(f"\nFEATURE-TARGET ALIGNMENT:")
    print(f"  Feature matrix patients: {len(ml_features_enhanced)}")
    print(f"  Target variable patients: {len(target_variables)}")
    print(f"  Patients in both: {len(set(ml_features_enhanced.index) & set(target_variables.index))}")
    
    if len(ml_features_enhanced) == len(target_variables):
        print("  Perfect alignment - ready for model training!")
    else:
        print("  Misalignment detected - check patient ID matching")
    
    # Create final dataset
    final_ml_dataset = create_final_dataset(ml_features_enhanced, target_variables)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING + FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    
    if len(final_ml_dataset) > 0:
        print("Successfully created ML-ready dataset!")
        print("Ready for model training and evaluation")
        print(f"Dataset size: {final_ml_dataset.shape[0]} patients x {final_ml_dataset.shape[1]} total columns")
        
        # Final data quality check
        missing_check = final_ml_dataset.isnull().sum().sum()
        if missing_check == 0:
            print("No missing values in final dataset")
        else:
            print(f"{missing_check} missing values detected in final dataset")
    else:
        print("Failed to create final dataset - check data alignment")
        final_ml_dataset = pd.DataFrame()
else:
    print("No enhanced features available for target preparation")
    final_ml_dataset = pd.DataFrame()

# ============================================================================
# STEP 9: MACHINE LEARNING MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: MACHINE LEARNING MODEL TRAINING AND EVALUATION")
print("=" * 80)

def fix_small_dataset_ml(X, y_reg, y_class):
    """Fixed ML pipeline specifically designed for small datasets"""
    
    print("=== DIAGNOSING ISSUES ===")
    print(f"Dataset size: {len(X)} patients")
    print(f"Features: {X.shape[1]} features")
    print(f"Feature-to-sample ratio: {X.shape[1]/len(X):.1f}:1")
    
    if X.shape[1] > len(X):
        print("MORE FEATURES THAN SAMPLES - Classic overfitting scenario!")
    
    print(f"Target variance: {y_reg.var():.2f}")
    print(f"Target range: {y_reg.min():.1f} - {y_reg.max():.1f} hours")
    
    # Issue 1: Feature selection
    print("\n=== FIX 1: FEATURE SELECTION ===")
    
    # Select only most important features
    n_features_to_select = min(6, len(X)//2)
    selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
    X_selected = pd.DataFrame(
        selector.fit_transform(X, y_reg),
        index=X.index,
        columns=X.columns[selector.get_support()]
    )
    
    print(f"Reduced features from {X.shape[1]} to {X_selected.shape[1]}")
    print("Selected features:")
    for i, feature in enumerate(X_selected.columns, 1):
        print(f"  {i}. {feature}")
    
    # Issue 2: Use regularized models
    print("\n=== FIX 2: REGULARIZED MODELS ===")
    
    models = {
        'Ridge_Regression': Ridge(alpha=1.0),
        'Lasso_Regression': Lasso(alpha=0.1),
        'Simple_RF': RandomForestRegressor(
            n_estimators=10,
            max_depth=2,
            min_samples_split=3,
            random_state=42
        )
    }
    
    # Issue 3: Use appropriate validation
    print("\n=== FIX 3: LEAVE-ONE-OUT VALIDATION ===")
    
    loo = LeaveOneOut()
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Store predictions for each fold
        y_true_loo = []
        y_pred_loo = []
        
        # Manual Leave-One-Out with proper indexing
        for train_idx, test_idx in loo.split(X_selected):
            X_train_loo = X_selected.iloc[train_idx]
            X_test_loo = X_selected.iloc[test_idx]
            y_train_loo = y_reg.iloc[train_idx]
            y_test_loo = y_reg.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_loo)
            X_test_scaled = scaler.transform(X_test_loo)
            
            # Train and predict
            model.fit(X_train_scaled, y_train_loo)
            pred = model.predict(X_test_scaled)[0]
            
            y_true_loo.append(y_test_loo.iloc[0])
            y_pred_loo.append(pred)
        
        # Calculate metrics
        y_true_loo = np.array(y_true_loo)
        y_pred_loo = np.array(y_pred_loo)
        
        mse = mean_squared_error(y_true_loo, y_pred_loo)
        mae = mean_absolute_error(y_true_loo, y_pred_loo)
        
        # Calculate R² manually
        ss_res = np.sum((y_true_loo - y_pred_loo) ** 2)
        ss_tot = np.sum((y_true_loo - np.mean(y_true_loo)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        results[name] = {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'R²': r2,
            'Predictions': y_pred_loo,
            'Actual': y_true_loo
        }
        
        print(f"  MSE: {mse:.1f}")
        print(f"  RMSE: {np.sqrt(mse):.1f} hours")
        print(f"  MAE: {mae:.1f} hours")
        print(f"  R²: {r2:.3f}")
    
    # Issue 4: Simplified classification
    print("\n=== FIX 4: SIMPLIFIED CLASSIFICATION ===")
    
    # Convert to binary classification
    y_binary = (y_reg > y_reg.median()).astype(int)
    print(f"Binary classification: Short stay (≤{y_reg.median():.1f}h) vs Long stay (>{y_reg.median():.1f}h)")
    print(f"Distribution: {np.bincount(y_binary)} [Short, Long]")
    
    if len(np.unique(y_binary)) == 2:
        log_reg = LogisticRegression(random_state=42, max_iter=1000)
        
        # LOO for classification
        y_true_class = []
        y_pred_class = []
        
        for train_idx, test_idx in loo.split(X_selected):
            X_train_loo = X_selected.iloc[train_idx]
            X_test_loo = X_selected.iloc[test_idx]
            y_train_loo = y_binary.iloc[train_idx]
            y_test_loo = y_binary.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_loo)
            X_test_scaled = scaler.transform(X_test_loo)
            
            log_reg.fit(X_train_scaled, y_train_loo)
            pred = log_reg.predict(X_test_scaled)[0]
            
            y_true_class.append(y_test_loo.iloc[0])
            y_pred_class.append(pred)
        
        accuracy = accuracy_score(y_true_class, y_pred_class)
        print(f"Binary Classification Accuracy: {accuracy:.3f}")
        
        results['Binary_Classification'] = {
            'Accuracy': accuracy,
            'Predictions': y_pred_class,
            'Actual': y_true_class
        }
    else:
        print("Cannot perform classification - insufficient class diversity")
    
    return results, X_selected

# Execute Step 9
if len(final_ml_dataset) > 0:
    print("=== STEP 1: PREPARING DATA FOR ML ===")
    
    # Use enhanced feature matrix as X
    X = ml_features_enhanced.copy()
    print(f"Features (X): {X.shape}")
    
    # Use target variables
    y_reg = target_variables['LOS_HOURS'].copy()
    y_class = target_variables['LOS_CATEGORY'].copy()
    
    print(f"Regression target (y_reg): {len(y_reg)} values")
    print(f"Classification target (y_class): {len(y_class)} values")
    
    # Ensure alignment
    common_patients = set(X.index) & set(y_reg.index)
    print(f"Common patients: {len(common_patients)}")
    
    if len(common_patients) > 0:
        # Filter to common patients
        X = X.loc[list(common_patients)]
        y_reg = y_reg.loc[list(common_patients)]
        y_class = y_class.loc[list(common_patients)]
        
        print(f"Final aligned dataset:")
        print(f"  X shape: {X.shape}")
        print(f"  y_reg length: {len(y_reg)}")
        print(f"  y_class length: {len(y_class)}")
        
        print("\nApplying fixes to dataset...")
        fixed_results, X_selected = fix_small_dataset_ml(X, y_reg, y_class)
        
        # Results interpretation
        print("\n" + "=" * 80)
        print("FIXED RESULTS - REALISTIC INTERPRETATION")
        print("=" * 80)
        
        print(f"\nCORRECTED DATASET:")
        print(f"  Original features: {X.shape[1]} → Selected features: {X_selected.shape[1]}")
        print(f"  Patients: {len(X)}")
        print(f"  Feature-to-sample ratio: {X_selected.shape[1]/len(X):.1f}:1 (improved)")
        
        print(f"\nREALISTIC REGRESSION RESULTS:")
        best_model = min(fixed_results.items(), key=lambda x: x[1].get('RMSE', float('inf')) if 'RMSE' in x[1] else float('inf'))
        
        for name, metrics in fixed_results.items():
            if 'RMSE' in metrics:
                print(f"\n  {name}:")
                print(f"    RMSE: {metrics['RMSE']:.1f} hours ({metrics['RMSE']/24:.1f} days)")
                print(f"    MAE: {metrics['MAE']:.1f} hours")
                print(f"    R²: {metrics['R²']:.3f}")
                
                if metrics['R²'] > 0:
                    print(f"    Model explains {metrics['R²']*100:.1f}% of variance")
                else:
                    print(f"    Model performs worse than baseline")
        
        print(f"\nBEST MODEL: {best_model[0]}")
        print(f"   RMSE: {best_model[1]['RMSE']:.1f} hours")
        
        # Show classification results if available
        if 'Binary_Classification' in fixed_results:
            print(f"\nBINARY CLASSIFICATION RESULTS:")
            print(f"   Accuracy: {fixed_results['Binary_Classification']['Accuracy']:.3f}")
            print(f"   Predictions vs Actual:")
            for i, (pred, actual) in enumerate(zip(fixed_results['Binary_Classification']['Predictions'],
                                                 fixed_results['Binary_Classification']['Actual'])):
                status = "CORRECT" if pred == actual else "INCORRECT"
                print(f"     Patient {i+1}: Predicted={pred}, Actual={actual} {status}")
        
        print(f"\nCLINICAL INTERPRETATION:")
        target_range = y_reg.max() - y_reg.min()
        best_rmse = best_model[1]['RMSE']
        relative_error = (best_rmse / target_range) * 100
        
        print(f"  Target range: {y_reg.min():.1f} - {y_reg.max():.1f} hours ({target_range:.1f}h span)")
        print(f"  Prediction error: ±{best_rmse:.1f} hours (±{best_rmse/24:.1f} days)")
        print(f"  Relative error: {relative_error:.1f}% of total range")
        
        if relative_error < 30:
            print(f"  Reasonable prediction accuracy for small dataset")
        elif relative_error < 50:
            print(f"  Moderate prediction accuracy - consider more data")
        else:
            print(f"  High prediction error - need significantly more data")
    
    else:
        print("No common patients found between features and targets!")
        print("Check data alignment issues.")

else:
    print("No final dataset available for machine learning")

print("\n" + "=" * 80)
print("ICU LENGTH OF STAY PREDICTION - ANALYSIS COMPLETE")
print("=" * 80)