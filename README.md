Self-Pruning Neural Network on CIFAR-10
📌 Project Overview

This project implements a Self-Pruning Neural Network trained on the CIFAR-10 dataset using PyTorch.
The goal is to study how L1 regularization (lambda) helps the network automatically reduce unnecessary weights (pruning) while maintaining good accuracy.

This experiment analyzes the trade-off between model sparsity and test accuracy.

🎯 Objectives
Train a CNN model on CIFAR-10
Apply L1 regularization with different lambda values
Measure:
Training Loss
Test Accuracy
Model Sparsity
Compare results for different pruning strengths

🗂️ Dataset
CIFAR-10 dataset
50,000 training images
10,000 test images
10 image classes

The dataset is automatically downloaded by PyTorch.

⚙️ Implementation Steps
Step 1 — Load CIFAR-10 dataset

Using torchvision.datasets.CIFAR10 with normalization and transforms.

Step 2 — Define CNN Model

A custom convolutional neural network is defined for image classification.

Step 3 — Add Self-Pruning (L1 Regularization)

L1 penalty is added to the loss:

Loss = CrossEntropyLoss + λ × L1_norm(weights)

This forces small weights to become zero → automatic pruning.

Step 4 — Train with Different Lambda Values

The model is trained for 10 epochs for:

λ = 1e-05
λ = 1e-04
λ = 1e-03
Step 5 — Evaluate Performance

For each lambda:

Average Training Loss
Test Accuracy
Sparsity Percentage

🧪 Experimental Results

| Lambda (λ) | Final Avg Loss | Test Accuracy | Sparsity |
| ---------- | -------------- | ------------- | -------- |
| 1e-05      | 0.7754         | **54.83%**    | 0.20%    |
| 1e-04      | 0.7757         | **55.51%**    | 0.20%    |
| 1e-03      | 0.7629         | **55.38%**    | 0.20%    |
