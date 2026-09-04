# DermaLens

An educational computer vision prototype for skin-image classification, model explainability, and cautious AI-generated guidance.

> **Important:** DermaLens is not a medical device and does not provide a medical diagnosis. Its predictions may be inaccurate or completely wrong. Any skin concern should be evaluated by a qualified medical professional.

## Overview

DermaLens is a personal machine learning project built to explore image classification, transfer learning, model explainability, and LLM-assisted communication.

I trained the included ResNet models using the DermNet dataset on a Kaggle GPU because local hardware was not suitable for deep-learning training.

The trained model is served through a Flask application where users can upload an image, inspect the predicted visual category, view a Grad-CAM attention map, and receive a cautious natural-language explanation.

## Core Features

- Skin-image classification across 23 broad DermNet categories
- ResNet50-based inference with a locally stored model checkpoint
- Grad-CAM visualization showing influential image regions
- Confidence score and top prediction analysis
- OpenAI-generated general safety guidance
- Explicit medical disclaimers and escalation advice
- Automatic CPU or CUDA device selection
- In-memory image processing without intentional file persistence
- Simple responsive Flask interface

## Processing Flow

1. The user uploads an image through the Flask interface.
2. OpenCV decodes and resizes the image.
3. PyTorch preprocessing normalizes it to the model input format.
4. ResNet50 produces probabilities across 23 categories.
5. Grad-CAM generates a heatmap from the final convolutional layer.
6. The heatmap is combined with the original image.
7. Only prediction labels and probabilities are sent to the OpenAI API.
8. The application displays the classification, visualization, and cautious guidance.

The original uploaded image is not sent to the OpenAI API.

## Model Details

| Property | Current implementation |
|---|---|
| Architecture | ResNet50 |
| Input size | 224 × 224 pixels |
| Dataset | DermNet |
| Number of classes | 23 |
| Framework | PyTorch |
| Explainability | Grad-CAM |
| Training environment | Kaggle GPU |
| Inference device | CUDA when available, otherwise CPU |

The repository also contains a ResNet18 checkpoint from model experimentation, while the current Flask inference flow uses ResNet50.

## Technology Stack

- Python
- Flask and Jinja
- PyTorch and Torchvision
- OpenCV
- Pillow
- NumPy
- Grad-CAM
- OpenAI API
- Gunicorn

## Project Structure

| Path | Responsibility |
|---|---|
| `app.py` | Flask routes, model loading, inference, Grad-CAM, and AI guidance |
| `templates/index.html` | Upload interface and result presentation |
| `models/class_names.json` | Classification category names |
| `models/best_resnet50_dermnet.pth` | Active ResNet50 checkpoint |
| `models/best_resnet18_dermnet.pth` | Experimental ResNet18 checkpoint |
| `examples/` | Images used during local experimentation |
| `requirements.txt` | Python dependencies |
| `Procfile` | Gunicorn process definition |

## Responsible AI Boundaries

DermaLens intentionally avoids presenting its output as a diagnosis.

The AI guidance is instructed to:

- communicate uncertainty clearly;
- recommend an in-person medical evaluation;
- avoid medication or dosage recommendations;
- avoid claiming that a condition is harmless;
- recommend urgent care for severe pain, bleeding, or rapid changes.

These safeguards reduce risk but do not make the system suitable for clinical use.

## Local Requirements

Local execution requires Python 3.9, the dependencies listed in `requirements.txt`, the bundled model checkpoint, and an `OPENAI_API_KEY` supplied through a local environment file.

Secrets and virtual environments should remain outside version control.

## Current Limitations

- No clinical validation has been performed
- Training and evaluation notebooks are not included
- No published accuracy, precision, recall, or confusion matrix
- Predictions are limited by the dataset and training distribution
- The application has no automated test suite
- Upload protection and production security controls are incomplete
- Generated guidance depends on an external API
- Dataset and example-image licensing should be reviewed before reuse
- The project is not deployed and is maintained as an educational prototype

## What I Learned

This project gave me practical experience with:

- training deep-learning models using cloud GPU resources;
- transfer learning with ResNet architectures;
- saving and loading PyTorch checkpoints;
- building an image preprocessing and inference pipeline;
- implementing Grad-CAM explainability;
- serving an ML model through Flask;
- integrating model output with an LLM;
- communicating uncertainty in a sensitive medical domain;
- separating image processing from external API communication.

## Project Status

DermaLens is an archived educational prototype. It is not deployed, actively operated, or intended for real medical decision-making.

## Author

**Roman Mammadov**

Built as a personal machine learning and backend engineering project.
