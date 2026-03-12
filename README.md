# DAU-KDAH: Dysarthric Multi-Lingual and Multimodal Speech Corpora for Indic Languages

[![Conference](https://img.shields.io/badge/APSIPA%20ASC-2025-blue)](https://www.apsipa.org/)
[![License](https://img.shields.io/badge/License-Research%20Use-green)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![Funding](https://img.shields.io/badge/Funded%20by-MeitY%20BHASHINI-purple)](https://bhashini.gov.in/)

> **Official repository** for the paper:  
> *"DAU-KDAH Dysarthric Multi-Lingual and Multimodal Speech Corpora for Indic Languages"*  
> Arth J. Shah\*, Hiya Chaudhari\*, Kavya Kumar, Arushi Srivastava, Priya J. Kaple, RavindraKumar M. Purohit, Dharmendra H. Vaghera, Bhavna Singh, Aparna Walanj, Abhishek Srivastava, and Hemant A. Patil  
> *2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*  
> DOI: [10.1109/APSIPAASC65261.2025.11249015](https://doi.org/10.1109/APSIPAASC65261.2025.11249015)

---

## Overview

Dysarthria is a motor speech disorder that impairs muscular control over articulation, resulting in degraded speech clarity, irregular rhythm, and poor intelligibility. Existing dysarthric speech corpora — such as UA-Speech and TORGO — are predominantly English-only and limited in linguistic scope, which severely restricts the development of inclusive Automatic Speech Recognition (ASR) systems for linguistically diverse populations.

This repository accompanies the **DAU-KDAH dataset**, a first-of-its-kind multilingual and multimodal dysarthric speech corpus for Indic languages, collected through a synergistic collaboration between:

- **Speech Research Lab, Dhirubhai Ambani University (DAU)**, Gandhinagar, Gujarat
- **Kokilaben Dhirubhai Ambani Hospital – Medical Research Institute (KDAH-MRI)**, Mumbai, Maharashtra

The dataset covers **Hindi**, **Marathi**, **Gujarati**, and **Indian English**, with speech recorded from both dysarthric patients (across low, medium, and high severity levels) and neurotypical control subjects.

---

## Repository Contents

```
.
├── DAU-KDAH_bilstm.ipynb                          # BiLSTM-based severity classification notebook
├── requirements.txt                               # Python dependency specifications
└── DAU-KDAH_Dysarthic_Multi-Lingual_and_...pdf   # Published conference paper (APSIPA ASC 2025)
```

---

## Dataset Description

### Corpus Statistics

| Parameter                     | DAU-KDAH      | TORGO   | UA-Speech |
|-------------------------------|---------------|---------|-----------|
| Total Files                   | 307           | 1,982   | 3,573     |
| Total Languages               | 4             | 1       | 1         |
| Total Duration (mins)         | 103.50        | 133.55  | 182.68    |
| Avg. Duration per Language    | 25 mins       | 5 mins  | 19 mins   |
| Total Dysarthric Speakers     | 25            | 5       | 19        |
| Total Size (MB)               | 970           | 244     | 334       |
| Sampling Frequency (Hz)       | 16,000        | 16,000  | 44,100    |
| Bit Resolution (bits)         | 16            | 16      | 16        |

### Severity-Level Classification

Speech samples are organized into four severity classes, following clinical evaluation guidelines:

| Class      | Description                                                                 | # Audio Clips (30s each) |
|------------|-----------------------------------------------------------------------------|--------------------------|
| **Normal** | Healthy speakers; regular articulation, pitch (F0), tone, and loudness      | 10                       |
| **Low**    | Mostly intelligible speech with slight rhythm changes or slurred words      | 10                       |
| **Medium** | Inconsistent prosody, uneven articulation, slower speaking rate             | 17                       |
| **High**   | Severely distorted speech, poor articulation, inconsistent F0 and loudness | 6                        |

### Recording Protocol

- **Equipment:** Three Zoom H4n Handy recorders (140 dB SPL, −120 dB EIN noise floor), paired with DSLR cameras placed at 90° and 180° relative to the subject
- **Multimodal capture:** Simultaneous audio-video recording to support future lip-to-wave synthesis research
- **Environment:** Controlled hospital room recordings at KDAH-MRI (6th floor, ~18 m above sea level), with noise reduction post-processing applied
- **Data collection phases:** 5 structured phases — 2 for normal baselines, 3 for dysarthric severity levels
- **Ethics clearance:** Obtained from the Institutional Ethics Committee (IEC) of both DAU and KDAH-MRI

---

## Baseline Experiments

### Features Evaluated

**Handcrafted Acoustic Features:**
- **MFCC** — Mel-Frequency Cepstral Coefficients (mel-scale filterbank)
- **LFCC** — Linear-Frequency Cepstral Coefficients (linear filterbank; captures higher-frequency deviations)
- **GFCC** — Gammatone-Frequency Cepstral Coefficients (cochlea-inspired; robust to noise)

**Self-Supervised / Transformer-Based Embeddings:**
- **HuBERT** — Hidden-Unit BERT; masked prediction pre-training on raw audio
- **Wav2Vec 2.0** — Self-supervised contextual speech representations
- **XLSR** — Cross-Lingual Speech Representations; multilingual Wav2Vec 2.0
- **Whisper** — Intermediate encoder embeddings from OpenAI's multilingual ASR model

### Classifiers

- **BiLSTM** — Bidirectional Long Short-Term Memory; captures temporal dependencies in both past and future context
- **CNN** — Convolutional Neural Network; extracts local time-frequency spatial patterns from spectrogram inputs

### Key Results (4-class severity classification)

| Feature  | BiLSTM Accuracy (%) | CNN Accuracy (%) |
|----------|---------------------|------------------|
| GFCC     | **60.46**           | 51.17            |
| MFCC     | 48.84               | 46.52            |
| LFCC     | 51.16               | 44.18            |
| HuBERT   | 37.21               | 44.18 (XLSR)     |
| Wav2Vec  | 46.51               | 47.52            |
| Whisper  | 51.16               | 44.18            |
| XLSR     | 58.13               | 58.13            |

> **Finding:** Traditional spectral features (MFCC, GFCC) perform better with BiLSTM, while transformer-based embeddings excel with CNN. The in-the-wild hospital recording environment is a primary factor in the lower absolute accuracies compared to studio-collected corpora (TORGO: 96.48%, UA-Speech: 93.78%).

### Latency Analysis

A key contribution of this work is the **latency period analysis** — evaluating the minimum audio duration required for reliable severity classification. Results show that the model achieves comparable performance at just **0.265 seconds** of audio relative to 8 seconds, demonstrating that the extracted features are stable, dysarthria-specific representations suitable for real-time assistive applications.

---

## Getting Started

### Prerequisites

Install all dependencies from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes the following packages:

| Group                        | Package          | Min. Version |
|------------------------------|------------------|--------------|
| **Deep Learning**            | torch            | ≥ 2.0.0      |
|                              | torchvision      | ≥ 0.15.0     |
|                              | torchaudio       | ≥ 2.0.0      |
| **Transformer Models**       | transformers     | ≥ 4.36.0     |
|                              | datasets         | ≥ 2.16.0     |
| **Scientific Computing**     | numpy            | ≥ 1.24.0     |
|                              | scipy            | ≥ 1.11.0     |
| **Visualisation**            | matplotlib       | ≥ 3.7.0      |
|                              | seaborn          | ≥ 0.13.0     |
| **ML Utilities**             | scikit-learn     | ≥ 1.3.0      |

> **GPU Support:** The `requirements.txt` installs the CPU-only build of PyTorch by default. For CUDA support, install PyTorch separately **before** running `pip install -r requirements.txt`:
> ```bash
> # Example for CUDA 11.8
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```
> Find the correct command for your CUDA version at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

> **Reproducible environments:** To freeze exact versions after a successful install, run:
> ```bash
> pip freeze > requirements_frozen.txt
> ```

### Dataset Access

The DAU-KDAH dataset is available upon request. Please contact the corresponding author with your institutional affiliation and intended use case:

**Prof. Hemant A. Patil**  
Speech Research Lab, DA-IICT, Gandhinagar, Gujarat, India  
📧 `hemant_patil@daiict.ac.in`

> Note: Dataset access is subject to the ethics committee guidelines of DAU and KDAH-MRI. The data may only be used for non-commercial academic research purposes.

### Directory Structure (Expected)

Once you have obtained the dataset, organize the MFCC features as follows:

```
dysarthricdataset/
└── MFCC/
    └── MFCC/
        ├── train/
        │   ├── high/
        │   ├── medium/
        │   ├── low/
        │   └── normal/
        ├── test/
        │   ├── high/
        │   ├── medium/
        │   ├── low/
        │   └── normal/
        └── val/
            ├── high/
            ├── medium/
            ├── low/
            └── normal/
```

Each subfolder contains `.mat` files with pre-extracted features stored under the key `'final'`.

### Running the BiLSTM Baseline

The provided notebook `DAU-KDAH_bilstm.ipynb` implements the full BiLSTM classification pipeline.

**Hyperparameters:**

| Parameter      | Value  |
|----------------|--------|
| Input Size     | 20 (MFCC coefficients) |
| Hidden Size    | 256    |
| Num Layers     | 2      |
| Num Classes    | 4      |
| Batch Size     | 64     |
| Learning Rate  | 0.003  |
| Epochs         | 25     |
| Dropout        | 0.255  |
| Optimizer      | Adam   |
| Loss Function  | CrossEntropyLoss |

**Steps:**

1. Set the dataset paths in the notebook:
   ```python
   train_data_path = "/path/to/MFCC/train"
   test_data_path  = "/path/to/MFCC/test"
   validation_data_path = "/path/to/MFCC/val"
   ```
2. Run all cells sequentially. The best-performing model is saved as `best_accuracy_model_BiLSTM.pth`.
3. Evaluation metrics (accuracy, confusion matrix, F1-score) are computed on the held-out test set.

The notebook was originally executed on **Kaggle** with GPU acceleration (`cuda:0`). It is recommended to run on a CUDA-capable GPU for reasonable training time.

---

## Reproducibility Notes

- The data loader (`PtDataset`) reads `.mat` files and pads all sequences to a maximum length of **600 frames**.
- Random shuffling is applied at dataset initialization; set a fixed seed for reproducibility:
  ```python
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  ```
- The model checkpoint is selected based on best **validation accuracy** across 25 epochs.

---

## Citation

If you use the DAU-KDAH dataset, the baseline code, or findings from this work in your research, please cite:

```bibtex
@inproceedings{shah2025daukdah,
  title     = {DAU-KDAH Dysarthric Multi-Lingual and Multimodal Speech Corpora for Indic Languages},
  author    = {Shah, Arth J. and Chaudhari, Hiya and Kumar, Kavya and Srivastava, Arushi and
               Kaple, Priya J. and Purohit, RavindraKumar M. and Vaghera, Dharmendra H. and
               Singh, Bhavna and Walanj, Aparna and Srivastava, Abhishek and Patil, Hemant A.},
  booktitle = {2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  pages     = {861--866},
  year      = {2025},
  doi       = {10.1109/APSIPAASC65261.2025.11249015},
  publisher = {IEEE}
}
```

---

## Funding

This research was funded by the **Ministry of Electronics and Information Technology (MeitY), Government of India**, under the **BHASHINI** project (Grant ID: 11(1)2022-HCC(TDIL)).

---

## Related Resources

- [UA-Speech Corpus](http://www.isle.illinois.edu/sst/data/UASpeech/)
- [TORGO Database](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
- [HuBERT (Hsu et al., 2021)](https://arxiv.org/abs/2106.07447)
- [Wav2Vec 2.0 (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [XLSR / XLS-R (Babu et al., 2022)](https://arxiv.org/abs/2111.09296)
- [Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)

---

## License

This repository is released for **academic and non-commercial research use only**, in accordance with the ethical guidelines approved by the Institutional Ethics Committees of Dhirubhai Ambani University and Kokilaben Dhirubhai Ambani Hospital. Redistribution of the dataset or any derived materials requires written permission from the authors.

---

*\* Equal contribution*
