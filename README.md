# FinNER-Data-Aug

A data augmentation toolkit for Financial Named Entity Recognition (NER) tasks.

## Overview

This study aims to develop a specialized Named Entity Recognition (NER) system tailored to financial documents. By employing a CNN-BiLSTM-CRF model
and applying FinBERT to augment data through contextual synonym replacement, our system can effectively identify key financial entities. Additionally, we demonstrate that CNN-BiLSTM-CRF model alone, when properly fine-tuned and data augmented, achieves robust performance and outperforms transformer-based models like SecBERT and BERT-Base-NER in financial NER tasks. Our experiment results indicate that the FinBERT-driven data augmentation approach significantly improves entity recognition accuracy in financial texts, providing a reliable and efficient alternative to more computationally intensive models.

## Features

- Data augmentation strategies for financial text
- Support for common financial named entities (e.g., company names, currencies, amounts)
- Easy-to-use interfaces for text transformation
- Configurable augmentation parameters
- Integration with multiple financial domain models:
  - FinBERT: BERT-based model pre-trained on financial text
  - SECBERT: Specialized model for SEC filings and financial reports
  - CNN-BiLSTM: Hybrid architecture combining CNN and BiLSTM for financial NER

## Installation
```bash
git clone https://github.com/yourusername/FinNER-Data-Aug.git
cd FinNER-Data-Aug
```


```bibtex
@software{finner_data_aug,
title = {FinNER-Data-Aug: A Data Augmentation Toolkit for Financial NER},
author = {Alex Yu},
year = {2024},
url = {https://github.com/alexyujiuqiao/FinNER-Data-Aug}
}
