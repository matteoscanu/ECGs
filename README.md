# ECG Classification with Time-Series Forecasting and Wavelet Scattering Transform

This project implements a complete system for cardiac arrhythmia classification using ECG signals from the MIT-BIH database. The system combines advanced data augmentation techniques based on time-series forecasting with Wavelet Scattering Transform (WST) for robust feature extraction, followed by k-Nearest Neighbors (kNN) classification.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Complete Pipeline](#complete-pipeline)
4. [Python Files - Beat Generation](#python-files---beat-generation)
5. [MATLAB Files - Pre-processing and Classification](#matlab-files---pre-processing-and-classification)
6. [Repository Usage](#repository-usage)
7. [File Structure](#file-structure)
---

## Overview

This project implements a complete system for ECG signal classification from the MIT-BIH Arrhythmia Database. The pipeline is divided into two main phases:

1. **Synthetic heartbeat generation** using time-series forecasting methods, developed in Python;
2. **Pre-processing, WST transformation, and classification** of ECG signals, developed in MATLAB.

The considered arrhythmia classes are:
- **N**: Normal beats
- **S**: Supraventricular premature beats
- **V**: Ventricular escape beats
- **F**: Fusion beats
---

## Requirements

### Python
```
numpy
pandas
scipy
matplotlib
wfdb
sktime
frechetdist
tslearn
scikit-learn
```

### MATLAB
- Wavelet Toolbox
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox
- WFDB Toolbox for MATLAB
---

## Complete Pipeline

### Phase 1: Beat Generation (Python)
```
gen_beats_*.py → beats_*.mat
```

### Phase 2: Classification (MATLAB)
```
main_script.m → Load beats_*.mat → Pre-processing → WST → Classification
```

---

## Python Files - Beat Generation

### Generation Files

Four Python files implement the same pipeline with **different forecasters**:

#### 1. `gen_beats_ets.py` - Exponential Smoothing
- **Forecaster**: `AutoETS` (Error-Trend-Seasonal)
- **Configuration**: `error="add"`, `seasonal="add"`, `sp=60` (pre_window + post_window)
- **Output**: `beats_ets.mat`
- **Metrics**: Also includes metric calculations for the **naive** method

#### 2. `gen_beats_arima.py` - ARIMA
- **Forecaster**: `AutoARIMA`
- **Output**: `beats_arima.mat`

#### 3. `gen_beats_lstm.py` - LSTM
- **Forecaster**: LSTM neural network
- **Output**: `beats_lstm.mat`

#### 4. `gen_beats_rnn.py` - RNN
- **Forecaster**: Recurrent neural network (RNN)
- **Output**: `beats_rnn.mat`

### Common Functionality

All files follow the same logic:

1. **Dataset Loading**: Imports the MIT-BIH Arrhythmia Database excluding files with pacemakers
2. **Class Mapping**: Converts original annotations to AAMI standard classes
3. **Peak-Rest Separation**: 
   - Isolates the R-peak (25 points before + 35 after = 60 points)
   - Separates the rest of the beat (remaining 140 points)
4. **Training**: The forecaster is trained on the peak portion
5. **Prediction**: Generates new synthetic R-peaks
6. **Recomposition**: Combines the generated peak with the rest of the beat by adding:
   - Gaussian noise to the rest of the beat
   - Jump correction to ensure continuity
7. **Metrics Calculation**: For each generated beat calculates:
   - **DTW** (Dynamic Time Warping)
   - **Wasserstein Distance**
   - **Cosine Similarity**
   - **Fréchet Distance**

### Key Parameters

```python
pre = 100              # Points before R-peak
post = 100             # Points after R-peak
pre_window = 25        # Pre-peak window to generate
post_window = 35       # Post-peak window to generate
sampling_ets = 5       # Number of samples per class
```

### Generated Classes

- **V**: 5 beats per iteration (25 total)
- **F**: 10 beats per iteration (50 total)
- **S**: 10 beats per iteration (50 total)

### Output

Each file generates:
- A `.mat` file containing a DataFrame with:
  - `Lead_I`: Array of generated beats
  - `Class`: Class label
  - `Augmented`: Flag '1' (all augmented)
- An `output_*.txt` file with average metrics
- Comparison plots between generated and real beats (.jpg format)

### Important Note: Naive Metrics

**Metrics for the naive method are calculated exclusively in the `gen_beats_ets.py` file**, where a "fake" beat is generated through simple Gaussian noise addition and compared with the real beat.

---

## MATLAB Files - Pre-processing and Classification

### Main File: `main_script.m`

This is the central file of the MATLAB pipeline. It performs all pre-processing, transformation, and classification operations.

#### Script Sections

##### 1. Dataset Download (Optional)
```matlab
% Commented block for automatic MIT-BIH Dataset download
% Includes WFDB Toolbox installation
```

##### 2. Import and Pre-processing
```matlab
% Loads all 48 MIT-BIH files
% Excludes files with pacemakers (102, 104, 107, 217)
% Extracts beats centered on R-peak (199 + 1 points)
% Maps classes according to AAMI standard
```

**Key variables:**
- `pre = 99`: Points before R-peak
- `post = 100`: Points after R-peak
- `sample_points = 200`: Total beat length

##### 3. Q-Class Removal
```matlab
% Removes all 'Q' type beats (unclassifiable)
```

##### 4A. Option 1 - Naive Augmentation (Commented)
```matlab
% If using naive augmentation:
% - Completely implemented in MATLAB
% - Calls the naive_augmentation.m function
% - Generates beats by adding Gaussian noise
```

##### 4B. Option 2 - Time-Series Forecasting (Default)
```matlab
% Loads beats generated from Python files
load('beats_rnn.mat')  % Change here to use ets, arima, lstm, or rnn
```

**IMPORTANT**: Modify the filename based on the forecaster used:
- `beats_ets.mat`
- `beats_arima.mat`
- `beats_lstm.mat`
- `beats_rnn.mat`

##### 5. Dataset Balancing
```matlab
% Uses the balance() function to:
% - Equalize the number of samples per class
% - Create a balanced dataset
% - Randomly organize beats with organize()
```

##### 6. Wavelet Scattering Transform (WST)
```matlab
% WST parameters:
invariance = 0.5       % Invariance scale
Q = [8 1]             % Quality factors
```

**What WST does:**
- Transforms each beat (200 points) into a matrix (paths × time_windows)
- Reduces dimensionality by selecting the time window with maximum peak
- Extracts robust and invariant features

##### 7. Classification with KNN

**4-class classification (N, V, S, F):**
```matlab
% k-fold cross-validation (k=10)
% KNN with k=4 neighbors
% Distance: euclidean
% DistanceWeight: inverse
```

**3-class classification (V, S, F):**
```matlab
% Same procedure but excludes 'N' class
% Useful for evaluating performance on arrhythmias only
```

##### 8. Visualizations
The script automatically generates:
- Individual beat plots
- WST filterbank
- Scaling function and wavelets
- Coefficient scalograms (0th, 1st, 2nd order)

---

### MATLAB Support Files

#### `naive_augmentation.m`
**Function for naive data augmentation**

```matlab
function [augmented_I, augmented_ann, augmented] = 
    augmentation(Lead_I, annotations, class_label, target_count, sample_points)
```

**Input:**
- `Lead_I`: ECG data matrix
- `annotations`: Label vector
- `class_label`: Class to augment
- `target_count`: Target number of beats to generate
- `sample_points`: Beat length

**Output:**
- `augmented_I`: New augmented ECG data
- `augmented_ann`: New labels
- `augmented`: Vector tracking which beats are augmented

**Operation:**
1. Selects all beats of the specified class
2. Generates `num_to_generate = target_count - current_count` new beats
3. For each new beat:
   - Randomly selects an original beat
   - Adds Gaussian noise: `noise ~ N(0, √0.05)`
   - Creates augmented beat: `augmented = original + noise`

**Note**: This technique is completely implemented in MATLAB, unlike forecasting methods that require Python.

#### `balance.m`
**Function for class balancing**

```matlab
function [I_balanced, ann_balanced, aug_balanced] = 
    balance(I, annotations, augmented, class_label, target_count)
```

**Purpose**: Samples exactly `target_count` beats for each class, randomly selecting from available beats (original + augmented).

#### `organize.m`
**Function for randomizing beat order**

```matlab
function [I_org, ann_org, aug_org] = 
    organize(I, annotations, augmented, seed)
```

**Purpose**: Randomly shuffles beats to facilitate fold creation for cross-validation.

#### `classCounter.m`
**Utility function for counting classes**

```matlab
function classCounter(annotations)
```

**Purpose**: Prints the number of beats for each class present in the dataset.

#### `classificationReport.m`
**Function for generating classification report**

```matlab
function classificationReport(y_test, y_pred, classes)
```

**Purpose**: Calculates and displays:
- Overall accuracy
- Precision, Recall, F1-score per class
- Confusion matrix

---

### Additional Python File

#### `histogram.py`
**Utility for visualizing class distribution**

```python
# Generates histograms to visualize 
# class imbalance in the dataset
```

---

## Repository Usage

### Complete Workflow

#### Step 1: Beat Generation with Time-Series Forecasting

Choose one of the four methods and execute the corresponding file:

```bash
# Option 1: Exponential Smoothing (includes naive metrics)
python gen_beats_ets.py

# Option 2: ARIMA
python gen_beats_arima.py

# Option 3: LSTM
python gen_beats_lstm.py

# Option 4: RNN
python gen_beats_rnn.py
```

**Produced output:**
- `beats_[method].mat`: MATLAB file with generated beats
- `output_[method].txt`: Evaluation metrics
- `.jpg` files: Comparison plots

**Execution time**: 5-15 minutes per method (depends on forecaster)

---

#### Step 2: Pre-processing and Classification in MATLAB

1. **Modify `main_script.m`** at the loading line:
   ```matlab
   load('beats_ets.mat')  % Change with the generated file
   ```

2. **Execute the script**:
   ```matlab
   main_script
   ```

3. **Output:**
   - Classification report for 4 classes
   - Classification report for 3 classes
   - `.mat` files with transformed dataset (optional)
   - Visualizations saved as `.jpg`

**Execution time**: 10-20 minutes (includes WST transformation)

---

### Alternative Workflow: Naive Augmentation

If you prefer to use only naive augmentation (without Python):

1. **Modify `main_script.m`**:
   ```matlab
   % COMMENT this block:
   load('beats_rnn.mat')
   % ...

   % UNCOMMENT this block:
   num_per_class = int(total / length(classes))
   [I_V, annotations_V, augmented_V] = augmentation(I, annotations, 'V', num_per_class, sample_points)
   [I_S, annotations_S, augmented_S] = augmentation(I, annotations, 'S', num_per_class, sample_points)
   [I_F, annotations_F, augmented_F] = augmentation(I, annotations, 'F', num_per_class, sample_points)
   ```

2. **Execute the script**:
   ```matlab
   main_script
   ```

**Advantages**: Faster, all in MATLAB  
**Disadvantages**: Potentially lower performance compared to forecasting methods

---

## File Structure

```
ECGs/
│
├── README.md                      # This file
│
├── PYTHON - Beat Generation
│   ├── gen_beats_ets.py           # Exponential Smoothing + naive metrics
│   ├── gen_beats_arima.py         # AutoARIMA
│   ├── gen_beats_lstm.py          # LSTM
│   ├── gen_beats_rnn.py           # RNN
│   └── histogram.py               # Visualization utility
│
├── MATLAB - Pre-processing and Classification
│   ├── main_script.m              # Main script
│   ├── naive_augmentation.m       # Naive data augmentation
│   ├── balance.m                  # Class balancing
│   ├── organize.m                 # Order randomization
│   ├── classCounter.m             # Class counting
│   └── classificationReport.m     # Metrics report
│
└── OUTPUT (generated during execution)
    ├── beats_*.mat                # Generated beats (Python → MATLAB)
    ├── output_*.txt               # Evaluation metrics
    ├── dataset.mat                # Processed dataset (optional)
    ├── transformed.mat            # WST-transformed dataset (optional)
    └── *.jpg                      # Visualizations and plots
```

---

## Key Points

### Differences Between Generation Files
- Identical structure and logic
- Only difference: The forecaster used
- Metrics calculated in all files (DTW, Wasserstein, Cosine, Fréchet)
- Naive metrics: Only in `gen_beats_ets.py`

### Pipeline Exceptions
1. **Naive Augmentation**: Completely implemented in MATLAB (does not require Python)
2. **Files with Pacemaker**: Automatically excluded (102, 104, 107, 217)
3. **Q Class**: Removed during pre-processing

### Best Practices
- Execute Python files once, then reuse the generated `.mat` files
- Save the WST-transformed dataset to avoid recalculations
- Use k-fold cross-validation for robust results
- Compare performance across different forecasters

---

## Recommended Execution Order

### First Complete Execution
1. `python gen_beats_ets.py` → obtain `beats_ets.mat` + naive metrics
2. Modify `main_script.m`: `load('beats_ets.mat')`
3. `matlab -r main_script`
4. Analyze results in `output_ets.txt` and MATLAB report

### Method Comparison
5. `python gen_beats_arima.py`
6. Modify `main_script.m`: `load('beats_arima.mat')`
7. `matlab -r main_script`
8. Repeat for LSTM and RNN
9. Compare performance

---

## Additional Notes

- **Dataset**: MIT-BIH Arrhythmia Database (48 files, 360k beats)
- **Sampling Frequency**: 360 Hz
- **Beat Length**: 200 points (≈ 0.55 seconds)
- **Classifier**: k-Nearest Neighbors (KNN)
- **Cross-Validation**: 10-fold

---

## Project Objectives

1. **Intelligent Data Augmentation**: Generate realistic synthetic beats using time-series forecasting
2. **Robust Feature Extraction**: Use Wavelet Scattering Transform for invariant features
3. **Accurate Classification**: Distinguish between normal beats and arrhythmias (V, S, F)
4. **Methodological Comparison**: Evaluate different forecasters and augmentation techniques

---

## References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/
- AAMI EC57 Standard: Standard arrhythmia classes
- Wavelet Scattering Transform: Multi-scale transform for signal analysis

---

## Author

Matteo Scanu, Davide Carbone, Lamberto Rondoni
Repository: [github.com/matteoscanu/ECGs](https://github.com/matteoscanu/ECGs)
---

## License

Project developed for research and educational purposes.
---

**Last update**: February 2026
