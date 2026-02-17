# ECG Classification with Time-Series Forecasting and Wavelet Scattering Transform

This project implements a classification process for cardiac arrhythmia classification using ECG signals from the MIT-BIH database, combining it with advanced data augmentation techniques based on time-series forecasting with Wavelet Scattering Transform (WST) for robust feature extraction, followed by k-Nearest Neighbors (kNN) classification.

## Overview

This project is divided into two main phases:

1. **Synthetic heartbeat generation** using time-series forecasting methods, developed in Python;
2. **Pre-processing, WST transformation, and classification** of ECG signals, developed in MATLAB.

The considered arrhythmia classes are:
- **N**: Normal beats;
- **S**: Atrial premature and supraventricular premature beats;
- **V**: Premature ventricular contraction and ventricular escape beats;
- **F**: Fusion of normal and ventricular beats.
---

## Required packages

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

### Phase 1: Beats Generation (Python)

1. Choose a time-series forecasting method and use its respective file in folder 'forecasting' (gen_beats_*.py) to generate as many beats as needed. Remember that:
    - ETS method is currently the one which yields the best results;
    - LSTM and RNN needs a GPU to run.
    - all files are also used to measure how distant new beats are from the original ones;
    - these files have been written considering three classes of arrhythmia. If you need more you might need to change the code a bit;
    - this procedure is only needed for beats that present some kind of cardiac disease.
2. All files will give as output two files: one with generated beats (stored inside a .mat file) and one with all the measures (stored inside a .txt file).

Note that you can skip this whole part if you just want to classify beats generated with the naive augmentation method, since it's implemented in file ```main.m```, which calls function ```naive_augmentation.m``` (but you need to uncomment some parts of the code: I noted all parts that eventually need to be commented or uncommented in a case like that).


### Phase 2: Classification (MATLAB)
1. Load the file you have obtained in Phase 1 (skip if you want to use only the naive augmentation method).
2. You can comment and uncomment all the blocks of lines you need: as default the code is setup to use data generated with one of the forecasting techniques.
3. Launch file ```main.m```: it will execute pre-processing on the data, data augmentation process, transformation of the dataset using WST and classification of the beats using kNN.
---

## Python Files - Beat Generation

### Files overview

Four Python files implement the same pipeline with **different forecasters**:

#### 1. `gen_beats_ets.py` - Exponential Smoothing
- **Forecaster**: `AutoETS` (Error-Trend-Seasonal)
- **Configuration**: `error="add"`, `seasonal="add"`
- **Output**: `beats_ets.mat`
- **Metrics**: Also includes metric calculations for the **naive** method

#### 2. `gen_beats_arima.py` - ARIMA
- **Forecaster**: `AutoARIMA`
- **Output**: `beats_arima.mat`

#### 3. `gen_beats_lstm.py` - LSTM
- **Forecaster**: Long Short-Term Memory (LSTM) Neural Network
- **Output**: `beats_lstm.mat`

#### 4. `gen_beats_rnn.py` - RNN
- **Forecaster**: Recurrent Neural Network (RNN)
- **Output**: `beats_rnn.mat`

### Code round-up

All files follow the same logic:

1. **Dataset Loading**: Imports the MIT-BIH Arrhythmia Dataset excluding files related to patients with pacemakers;
2. **Class Mapping**: Converts original annotations to ANSI/AAMI EC57:1998 standard classes;
3. **Peak-Rest Separation**: Isolates the R-peak from the rest of the beat;
4. **Training**: The forecaster is trained on a certain amount of peaks (usually nine or ten);
5. **Prediction**: Generates new synthetic R-peaks;
6. **Recomposition**: Combines the generated peak with the rest of the beat by adding:
   - Gaussian noise to the rest of the beat;
   - Jump correction to ensure continuity;
7. **Metrics Calculation**: For each generated beat the following distances with the beats used to generate it are computed:
   - **DTW** (Dynamic Time Warping);
   - **Wasserstein Distance**;
   - **Cosine Similarity**;
   - **Fréchet Distance**.
8. Note that metrics for the naive augmentation method are calculated exclusively in the `gen_beats_ets.py` file, where a great amount of beats is generated through simple Gaussian noise addition and compared with the real ones.
---

### Output

Each file generates:
- A `.mat` file containing a DataFrame with:
  - `Lead_I`: Array of generated beats;
  - `Class`: Label class;
  - `Augmented`: Flag '1' (all augmented);
- An `output_*.txt` file with average metrics;
- Comparison plots between generated and real beats (.jpg format).

## MATLAB Files - Pre-processing and Classification

### Main File: `main.m`

This is the central file of the MATLAB pipeline. It performs pre-processing, transformation, and classification of given data.

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
% Extracts beats centered on R-peak (200 sample points each)
% Maps classes according to ANSI/AAMI EC57:1998 standard
```

##### 3. Q-Class Removal
```matlab
% Removes all 'Q' type beats (unclassifiable)
```

##### 4A. Option 1 - Naive Augmentation (Commented)
```matlab
% If using naive augmentation:
% - Completely implemented in MATLAB;
% - Calls the naive_augmentation.m function;
% - Generates beats by adding Gaussian noise.
```

##### 4B. Option 2 - Time-Series Forecasting (Default)
```matlab
% Loads beats generated from Python files
load('beats_*.mat')  % Change * with ets, arima, lstm, or rnn depending on the desired forecaster.
```

##### 5. Dataset Balancing
```matlab
% Uses the balance() function to:
% - Equalize the number of samples per class;
% - Create a balanced dataset;
% - Randomly organize beats with organize().
```

##### 6. Wavelet Scattering Transform (WST)
```matlab
% WST parameters:
invariance = 0.5 % Invariance scale
Q = [8 1]        % Quality factors
```

**WST OUTPUT:**
- Transforms each beat (stored in a vector) into a matrix (paths $\times$ time_windows);
- Dimensionality needs to be reduced: only one time window is taken, the one with maximum peak.

##### 7. Classification with KNN

**4-class classification (N, V, S, F):**
```matlab
% k-fold cross-validation (k=10)
% KNN with k=4 neighbors
% Each vote is computed with euclidean distance and an inverse distance weight
% (higher the distance, the less the value of the vote).
```

**3-class classification (V, S, F):**
```matlab
% Same procedure but excludes 'N' class;
% Useful for evaluating performance on arrhythmias only.
```

##### 8. Visualizations
The script automatically generates:
- Individual beat plots;
- WST filterbank;
- Scaling and wavelets functions;
- Coefficient scalograms (0th, 1st, 2nd order).
---

### MATLAB Support Files

#### `naive_augmentation.m`
**Function for naive data augmentation**

```matlab
function [augmented_I, augmented_ann, augmented] =
    augmentation(Lead_I, annotations, class_label, target_count, sample_points)
```

**Input:**
- `Lead_I`: ECG data matrix;
- `annotations`: Label vector;
- `class_label`: Class to augment;
- `target_count`: Target number of beats to generate;
- `sample_points`: Length of each beat.

**Output:**
- `augmented_I`: New augmented ECG data;
- `augmented_ann`: New labels;
- `augmented`: Vector tracking which beats are augmented.

**Operation:**
1. Selects all beats of the specified class;
2. Generates `num_to_generate = target_count - current_count` new beats;
3. For each new beat:
   - Randomly selects an original beat;
   - Adds Gaussian noise: $ w \sim N(0, \sqrt{0.05}) $
   - Creates augmented beat: `augmented = original + noised`

Remember that this technique is completely implemented in MATLAB, unlike forecasting methods that require Python.

#### `balance.m`
**Function for class balancing**

```matlab
function [I_balanced, ann_balanced, aug_balanced] =
    balance(I, annotations, augmented, class_label, target_count)
```

**Purpose**: Samples exactly `target_count` beats for each class, randomly selecting from the available ones.

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
- Accuracy, Precision, Sensitivity and Sensibility per class;
- Confusion matrix.

---

### Additional Python File

#### `histogram.py`
**Utility for visualizing class distribution**

```python
# Generates an histogram to visualize
# the final classification scores for
# each augmentation technique. 
```
---

## Files Structure

```
ECGs/
│
├── README.md                  # This file
├── main.m                     # Main script
├── naive_augmentation.m       # Naive data augmentation
├── balance.m                  # Class balancing
├── organize.m                 # Order randomization
├── classCounter.m             # Class counting
├── classificationReport.m     # Metrics report
├── histogram.py               # Visualization utility
│
├── forecasting (beats generator using time-series forecasting techniques)
│   ├── gen_beats_ets.py       # Exponential Smoothing + naive metrics
│   ├── gen_beats_arima.py     # AutoARIMA
│   ├── gen_beats_lstm.py      # LSTM
│   └── gen_beats_rnn.py       # RNN
│
└── OUTPUT (generated during execution)
    ├── beats_*.mat            # Generated beats
    ├── output_*.txt           # Evaluation metrics
    ├── dataset.mat            # If you need to save the dataset
    ├── transformed.mat        # WST-transformed dataset
    └── *.jpg                  # Visualizations and plots
```
---

## Additional Notes

- **Dataset**: MIT-BIH Arrhythmia Database (48 files)
- **Sampling Frequency**: 360 Hz
- **Beat Length**: 200 points ($\approx$ 0.56 seconds)
- **Classifier**: k-Nearest Neighbors (kNN)
- **Cross-Validation**: 10-fold
---

## References

If you use this code, please cite my thesis work! You can also use it to dive deep in the theory behind it.
```
@mastersthesis{scanu:tesi,
    author = {Matteo Scanu},
    title = {Synthesis of realistic human heart-beats with time-series forecasting techniques and arrhythmia classification using Wavelet Scattering Transform},
    school = {Politecnico di Torino},
    year = {2025}
}
```
---

## Author

Matteo Scanu
Repository: [github.com/matteoscanu/ECGs](https://github.com/matteoscanu/ECGs)
---

## License

Project developed for research and educational purposes.
For any doubt or curiosity, please
---

**Last update**: 17 February 2026
