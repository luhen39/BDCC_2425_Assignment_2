# ICU Length of Stay Prediction using MIMIC-III on Google Cloud

This project predicts the length of stay (LOS) in the ICU using patient data from the MIMIC-III database. It demonstrates how to handle large-scale medical data (~33GB), perform preprocessing using chunking and parallelism with **Dask**, train ML models, and deploy the full pipeline using Google Cloud Platform (GCP) resources.

---

## Project Objective

- Process and analyze large-scale clinical time-series data (CHARTEVENTS.csv)
- Engineer meaningful features from raw ICU data
- Predict ICU patient length of stay using regression models
- Utilize Google Cloud infrastructure for scalable data handling and model training

---

## Directory Structure

| File / Folder                 | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `Big_Data_Assignment_2_.ipynb` | Jupyter notebook covering **initial data exploration and analysis on a small sample**. For full dataset processing and modeling, see `big_data_assignment2_report.pdf`|
| `big_data_small_sample.py`    | Script for testing and analyzing a small portion of the dataset             |
| `big_data_final_version.py`   | Full pipeline for the complete dataset                                      |
| `big_data_results.txt`        | Output from `big_data_final_version.py` which is length of stay prediction on ICU |
|`big_data_assignment_2_report.pdf`| Final Assignment Report |
| `performance_report.txt`      | Summary of training and inference time and system usage                     |
| `plots/`                      | Visualizations generated during execution of `big_data_final_version.py`   |
| `profiling/`                  | Profiling logs for chunked data processing and model training               |
| `starting_script.sh`          | Setup script to prepare VM environment (install packages, configure RAID)   |

---

## Google Cloud Infrastructure Setup

### Google Cloud Storage
- Created a GCP storage bucket to store all necessary files including large dataset files, outputs, and logs.

### Virtual Machines

Two VMs were provisioned on Google Cloud:

1. **Basic VM for Data Transfer**
   - **Purpose**: Transfer `CHARTEVENTS.csv` (~33GB) from Google Drive to the GCP bucket.
   - **Tool used**: `rclone` for efficient Drive-to-Bucket upload.

2. **Powerful VM for Processing**
   - **Specs**:
     - 16 vCPUs
     - 64 GB RAM
     - Ubuntu 24.04
     - 100GB Persistent Disk
     - 2× 375GB Local SSDs (configured in RAID 0 for high-speed I/O)
   - **On startup**: ran `starting_script.sh` to:
     - Install required packages and Python libraries
     - Set up RAID 0 using `mdadm`
     - Mount local SSDs to `/mnt/disks/`

### File Operations
- Used `gsutil -m` for parallel file upload/download between storage and VM.

---

## Models and Methodology

### Data Preprocessing

- Used **Dask** to perform chunked and parallel processing of `CHARTEVENTS.csv`, enabling efficient handling of the full 33GB dataset without exhausting memory.
- Selected useful features (e.g., heart rate, BP, temperature)
- Merged with `ADMISSIONS.csv` and `ICUSTAYS.csv` for patient context
- Converted time to features and filled missing values with median
- Performed one-hot encoding where applicable
- Standardized numeric features

### Model Training and Evaluation

- **Train/test split**: 80/20
- **Validation method**: Standard train-test split (no cross-validation used in final evaluation)
- **Models evaluated**:
  - Ridge Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor

- **Hyperparameter tuning**: Conducted using `GridSearchCV` during experimentation phase

- **Performance Summary**:
  - **Ridge Regression**:
    - RMSE: 1253.5 hours (~52.2 days)
    - MAE: 81.4 hours
    - R²: -105.751 (severely underperforms baseline)
  - **Random Forest Regressor**:
    - RMSE: 115.6 hours (~4.8 days)
    - MAE: 62.5 hours
    - R²: 0.093 (explains 9.3% of variance)
  - **Gradient Boosting Regressor**:
    - RMSE: 115.9 hours (~4.8 days)
    - MAE: 62.5 hours
    - R²: 0.088 (explains 8.8% of variance)

- **Best Model**: **Random Forest Regressor**
  - Achieved the lowest RMSE and highest R² among all models tested.

- **Libraries Used**:
  - `scikit-learn`, `dask`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `psutil`, `gc`, `os`, `time`, `datetime`, `cProfile`, `pstats`, `functools`


### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Output and Performance

- Predictions and evaluation metrics saved in `big_data_results.txt`
- Plots saved to `plots/` (e.g., feature importance, prediction vs actual)
- Profiling logs:
  - `profiling/process_complete_dataset_in_chunks_profile.txt`: Dataset chunking performance
  - `profiling/train_and_evaluate_models_profile.txt`: Training phase profiling
- Performance Summary:
  - Summarized in `performance_report.txt`
  - Includes training time, CPU/RAM usage, disk I/O, and model size

---

Final Grade : 20/20
