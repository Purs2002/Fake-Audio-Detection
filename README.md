# Audio Deepfake Detection Using Acoustic Features and Machine Learning

## Project Overview

This project investigates the detection of AI-generated (fake) and genuine speech using acoustic feature extraction and machine learning techniques.

The **ADD 2022 (Audio Deepfake Detection 2022)** dataset is used to analyze differences between genuine and fake audio signals. The project explores multiple acoustic representations, including MFCC, LFCC, Constant-Q Transform (CQT), Jitter, and Shimmer.

Both supervised and unsupervised machine learning approaches are investigated to understand how effectively these acoustic features distinguish genuine speech from manipulated or synthesized speech.

## Objectives

- Analyze acoustic characteristics of genuine and fake speech.
- Extract relevant speech features from audio signals.
- Investigate the effectiveness of different acoustic feature representations.
- Apply clustering and anomaly detection techniques.
- Evaluate detection performance using Equal Error Rate (EER), Adjusted Rand Index (ARI), and Silhouette Score.
- Compare the performance of MFCC, LFCC, and CQT features.

## Dataset

**Dataset:** ADD 2022 (Audio Deepfake Detection)

The dataset contains genuine and fake audio samples.

The notebook uses the training and development subsets:

- Training dataset
- Development dataset

The dataset is loaded from Google Drive. The six split ZIP files are combined into a single archive and extracted for processing.

## Methodology

### 1. Dataset Preparation

- Mount Google Drive in Google Colab.
- Combine the six dataset ZIP parts.
- Extract the ADD 2022 dataset.
- Read training and development labels.
- Identify genuine and fake audio samples.

### 2. Acoustic Feature Extraction

The project investigates two groups of acoustic features.

#### A. Voice Quality Features

**Jitter**

Measures variations in the fundamental frequency period between consecutive vocal cycles.

**Shimmer**

Measures variations in the amplitude of consecutive vocal cycles.

These features are extracted using Praat through the Parselmouth Python library.

#### B. Spectral and Cepstral Features

**MFCC (Mel-Frequency Cepstral Coefficients)**

Represents the short-term spectral characteristics of speech using the Mel-frequency scale.

**LFCC (Linear-Frequency Cepstral Coefficients)**

Represents spectral characteristics using linear-frequency analysis.

**CQT (Constant-Q Transform)**

Provides a logarithmically spaced frequency representation of the audio signal.

For MFCC, LFCC, and CQT, statistical representations are generated using the mean and standard deviation of the extracted coefficients across time.

### 3. Feature Visualization

- Jitter versus Shimmer scatter plots.
- PCA-based two-dimensional visualization.
- Comparison of genuine and fake audio distributions.
- Visualization of K-Means clustering results.

### 4. Machine Learning Approaches

The notebook investigates the following approaches:

**A. Logistic Regression**

- Uses Jitter and Shimmer features.
- Applies StandardScaler.
- Uses balanced class weights.
- Generates prediction scores.
- Estimates Equal Error Rate (EER).

**B. K-Means Clustering**

- Uses MFCC, LFCC, and CQT separately.
- Performs two-cluster classification.
- Evaluates cluster separation and agreement with actual labels.

**C. One-Class Support Vector Machine (One-Class SVM)**

- Uses genuine audio samples for training.
- Learns the characteristics of genuine speech.
- Identifies anomalous audio samples.
- Uses MFCC, LFCC, and CQT independently.
- Estimates EER using ROC analysis.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| EER | Measures the operating point where false acceptance and false rejection rates are approximately equal. |
| Silhouette Score | Evaluates the separation and compactness of clusters. |
| Adjusted Rand Index (ARI) | Measures agreement between predicted clusters and actual labels. |
| ROC Curve | Evaluates detection performance across different thresholds. |

**Note:** EER results from the current exploratory experiments should not be interpreted as final generalization performance because the notebook evaluates some models on the same samples used during fitting.

## Technologies and Libraries

- Python
- Google Colab
- NumPy
- Pandas
- Librosa
- Praat Parselmouth
- Scikit-learn
- SciPy
- Matplotlib

## Project Workflow

```text
ADD 2022 Dataset
       |
       v
Dataset Preparation
       |
       v
Audio Signal Processing
       |
       v
Acoustic Feature Extraction
       |
       +-------------------------+
       |                         |
       v                         v
Jitter & Shimmer           MFCC, LFCC & CQT
       |                         |
       v                         v
Logistic Regression        K-Means Clustering
       |                         |
       |                         v
       |                    PCA Visualization
       |
       +-------------------------+
                                 |
                                 v
                         One-Class SVM
                                 |
                                 v
                         ROC Curve & EER
                                 |
                                 v
                         Performance Analysis
```

## Installation

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn scipy librosa praat-parselmouth
```

## Execution

1. Open the notebook in Google Colab.
2. Mount your Google Drive.
3. Upload or place the ADD 2022 dataset ZIP parts in the `ADD2022` directory.
4. Execute the dataset extraction cells.
5. Run the acoustic feature extraction procedures.
6. Execute the machine learning experiments.
7. Analyze the generated plots and evaluation metrics.

## Generated Outputs

The notebook generates the following CSV files:

- `ADD2022_jitter_shimmer_sample.csv`
- `ADD2022_mfcc_lfcc_cqt.csv`
- `ADD2022_separate_clustering_results.csv`
- `ADD2022_cluster_assignments.csv`

It also generates scatter plots, PCA visualizations, ROC curves, and evaluation results.

## Research Scope

This work explores the potential of acoustic features for distinguishing genuine speech from fake speech. It investigates both conventional voice-quality measurements and spectral representations, along with supervised classification, unsupervised clustering, and anomaly detection.

The experiments provide an exploratory foundation for developing more robust audio deepfake detection systems.

## Future Work

- Evaluate models on independent test data.
- Investigate feature fusion techniques.
- Explore advanced classifiers such as SVM, Random Forest, and deep learning architectures.
- Investigate CNN- and RNN-based audio representations.
- Perform cross-dataset evaluation.
- Improve robustness against unseen voice-generation techniques.

## Author

**Project:** Audio Deepfake Detection Using Acoustic Features and Machine Learning

**Domain:** Digital Signal Processing | Speech Processing | Machine Learning | Audio Forensics

---

*This repository contains exploratory research and implementation experiments using the ADD 2022 dataset.*
