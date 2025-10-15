# Clinical Text Classification: LoRA vs Full Finetuning

Comparison of parameter efficient LoRA finetuning against tradional full finetuining for medical transcription classification using DistilBERT

## Project Overview

This project classifies medical transcriptions obtained from the MTSamples dataset into 10 medical specialties, and compares the two finetuning strategies:

- **Full Finetuning**: Updates all 66M parameters of DistilBERT
- **LoRA**: Updates only 1.2M parameters (1.74%)

## Key Results

LoRA has acheived about 5% higher macro F1 score while training only less than 2% of the total number of trainable parameters compared to Full Finetuning, establishing its efficiency and competitive performative capabilities.

## Quick Start

### 1. Clone the Repository

git clone https://github.com/nathan-limjw/clinical-text-classification.git

cd clinical-text-classification

### 2. Installing Dependencies

python-m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

### 3. Downloading the Dataset

The dataset can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

The `mtsamples.csv` should then be placed under the `data/` directory

### 4. Running the Script

python main.py

This will: 
1. Preprocess the data (filtering for top 10 specialties, remove unnecessary entries, encode labels, split data, tokenize data)
2. Run Full Finetuning with Hyperparameter Search (8 Configs)
3. Run LoRA Finetuning with Hyperparameter Search (9 Configs)
4. Evaluate optimised models on test set
5. Generate classification report & confusion matrices

## Dataset

### MTSamples Dataset
- **Total Samples**: 4999 medical transcriptions
- **After Preprocessing**: 3614 samples (top 10 specialties)
- **Training Set**: 2529 samples (70%)
- **Validation Set**: 362 samples (10%)
- **Test Set**: 723 samples (20%)

## Methodology

### Model Architecture
- **Base Model**: DistilBERT-base-uncased (66M parameters)
- **Task**: 10-class sequence classification
- **Max Sequence Length**: 512 tokens
- **Tokenization**: WordPiece with paddintg and truncation

### Finetuning Approaches

#### Full Finetuning
- **Trainable Parameters**: 66M
- **Hyperparameter Search**:
    - Learning Rates: [1e-5, 2e-5, 3e-5, 5e-5]
    - Batch Sizes: [8,16]
- **Best Configuration**: LR = 5e-5, BS = 16
- **Training**: 10 epochs with early stopping (best at epoch 3)

#### LoRA Finetuning
- **Trainable Parameters**: 1.2M (1.74%)
- **Target Modules**: q_lin, v_lin (query and value projections)
- **Hyperparameter Search**:
    - Ranks: [8, 16, 32]
    - Learning Rates: [1e4, 2e4, 3e-4]
    - Alpha: 2 * rank
    - Dropout: 0.1
- **Best Configuration**: Rank = 32, LR = 1e-4
- **Training**: 10 epochs with early stopping (best at epoch 6)

### Training Configuration
- **Optimiser**: AdamW (weight decay = 0.01)
- **Learning Rate Schedule**: 500-step linear warmup
- **Early Stopping**: Based on validation macro F1-score
- **Hardware**: Google Colab with T4 GPU

## Results

### Final Test Performance

| Metric | Full Finetuning | LoRA | Difference|
| -------- | ------------ | ------| ----------- |
| Accuracy | 0.4578 | 0.4550 | -0.28% |
| Macro F1 | 0.3332 | **0.3498** | **+5.0%** |
| Weighted F1 | 0.4110 | **0.4326** | **+5.3%** |

### Training Dynamics

**Overftting Analysis:**
- Full Finetuning: Peaked at epoch 3 and declined 26.8% by epoch 10
- LoRA: Peaked at epoch 6, then declined 14.4% by epoch 10

LoRA demonstrated better stability and implicit regularization

### Confusion Matrices

Confusion matrices are under the `output/` directory

**Key Observations:**
- LoRA shows stronger diagonal (more correct predictions)
- Both models struggle with rare classes due to limited training data
- Common confusion between similar specialties

## Outputs

After training completes, results are saved in `output/`:

- `confusion_matrix_full.png`: Confusion matrix for full finetuning
- `confusion_matrix_lora.png`: Confusion matrix for LoRA finetuning
- `full_test_report.txt`: Detailed classification report for full finetuning
- `lora_test_report.txt`: Detailed classification report for LoRA finetuning

Models would be saved under `models`, but is not included in this repository due to size

## Limitations

- Only 10 out of 40 specialties used (oversimplification of task)
- DistilBERT is trained on general domain (not medical specific like BioBERT or ClinicalBERT)
- Limited hyperparameter search scope

**Note**: This is a coursework project for educational purposes, and should not be used in clinical deployment.