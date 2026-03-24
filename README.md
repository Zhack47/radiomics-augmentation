# Radiomics Augmentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/license/mit)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Welcome to **Radiomics Augmentation**, a pipeline design to help you apply data augmentation transforms to medical images in order to generate augmented radiomics feature sets.
Radiomics Augmentation was designed to combat class imbalance and small sample sizes typical in medical datasets.

## 📖 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Installation](#️-installation)
- [Quick Start](#-quick-start)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🎯 Overview
In radiomics studies, machine learning algorithms often struggle with high-dimensional feature spaces and small patient cohorts.
This repository provides an end-to-end framework to augment radiomics data, ensuring more robust, generalizable, and reproducible predictive models (e.g., for survival prediction). 

It integrates popular medical imaging frameworks such as `SimpleITK`, `torchio`, and `pyradiomics`.

## ✨ Features
* **Image-Level Perturbations:** Apply spatial transformations (rotation, translation, scaling) and domain-specific artifacts (noise injection, low-resolution simulation) to 3D medical images (CT, PET, MRI) and their corresponding segmentation masks prior to feature extraction.
* **Automated Pipeline:** Automated radiomic features extraction across augmented image variants.

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git https://github.com/Zhack47/radiomics-augmentation.git
cd radiomics-augmentation
pip install -r requirements.txt
```

Note: It is highly recommended to use a virtual environment (e.g., conda or venv).

## 🚀 Quick Start

An example demonstrating the use of Radiomics Augmentation on the HECKTOR2022 dataset is available in ```examples/Hecktor22.py```.

## 📄 License

This project is licensed under the MIT License

## 🙏 Contributors
Developed and maintained by @Zhack47.

Built to support robust medical imaging and radiomics research.