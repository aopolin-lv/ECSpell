# ECSpell
Code for paper "General and Domain Adaptive Chinese Spelling Check with Error Consistent Pretraining"

## 📖 Introduction
This repository contains the official PyTorch implementation for ECSpell. Our approach leverages error-consistent pretraining to improve generalization and domain adaptation for Chinese Spelling Check (CSC).

## 🛠️ Environment Setup
We recommend Python 3.8+ and PyTorch >= 1.9.0.
```bash
conda create -n ecspell python=3.8 -y
conda activate ecspell
pip install torch transformers jieba pypinyin Pinyin2Hanzi tqdm scipy
```

Install the local `glyce` library:
```bash
cd glyce
python setup.py develop
pip show glyce   # to ensure the successful installation of glyce lib
```

## 📁 Data Preparation
1. **Model Weights**: Download the pre-trained weights from [Google Drive](https://drive.google.com/file/d/1HlfDbMpXR6YHiBuJS8s_K3ZKG6j0fvc5/view?usp=sharing) and place them in `Transformers/glyce/`.
2. **Datasets**: Download SIGHAN or other CSC datasets. Organize them following the structure:
   - `Data/domains_data/` for domain-specific data.
   - `data/basedata/simplified/` or `data/basedata/traditional/` for baseline training data.
3. **Tokenization** (if required):
   ```bash
   python3 Code/tokenize_sequence.py
   ```

## 🏋️ Training
Set the Python path and run the training script:
```bash
export PYTHONPATH=$PYTHONPATH:./:./Code
python Code/train_baseline.py \
  --config_path Transformers/glyce_bert_both_font.json \
  --output_dir ./outputs
```
*(Adjust arguments in `train_baseline.py` as needed for your specific setup.)*

## 📊 Evaluation & Inference
- For evaluation, use the scripts in `csc_evaluation/` or modify the training script to enable evaluation mode.
- `csc_evaluation/build_train_data.py` can be used to construct and split training/validation datasets.

## 📝 Data Usage Notes
Path: `Data/domains_data`
- For zero-shot tasks, you should combine the `*.train` file and `*.test` file.
- For common tasks, the `*.train` file is used to do training and do evaluating while `*.test` is adopted to do predicting.

## 📜 Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{lv2023general,
  title={General and Domain-adaptive Chinese Spelling Check with Error-consistent Pretraining},
  author={Lv, Qi and Cao, Ziqiang and Geng, Lei and Ai, Chunhui and Yan, Xu and Fu, Guohong},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing},
  volume={22},
  number={5},
  pages={1--18},
  year={2023},
  publisher={ACM New York, NY}
}
```
