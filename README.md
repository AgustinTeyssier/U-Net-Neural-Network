# U-Net Neural Network for Pneumothorax Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A U-Net convolutional neural network implementation for pneumothorax segmentation in chest X-ray images.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a U-Net architecture for semantic segmentation of pneumothorax (collapsed lung) in chest X-ray images. The model identifies and segments pneumothorax regions using convolutional neural networks.

**Key Features:**
- U-Net architecture with customizable depth
- Data augmentation pipeline
- Dice coefficient and IoU evaluation metrics
- Pretrained model support (optional)
- Visualization tools for predictions

## Dataset

The model uses the **Pneumothorax Chest X-Ray Images and Masks** dataset from Kaggle:

- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks)
- **Contents:** Chest X-Ray images with corresponding pneumothorax masks
- **Images:** DICOM format with binary masks
- **Purpose:** Medical image segmentation for pneumothorax detection

### Dataset Structure
