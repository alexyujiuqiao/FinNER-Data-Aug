# FinNER-Data-Aug

A data augmentation toolkit for Financial Named Entity Recognition (NER) tasks.

## Overview

This project focuses on enhancing financial text datasets through data augmentation techniques to improve Named Entity Recognition models in the financial domain. The toolkit is designed to work with multiple financial domain models including FinBERT, SECBERT, and CNN-BiLSTM architectures.

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

```bibtex
@software{finner_data_aug,
title = {FinNER-Data-Aug: A Data Augmentation Toolkit for Financial NER},
author = {Alex Yu},
year = {2024},
url = {https://github.com/alexyujiuqiao/FinNER-Data-Aug}
}