# Voice Emotion Recognition with CNNs

**Group members:** Mateus BASTOS SOARES, Júlia Ellen DIAS LEITE, Helena GUACHALLA DE ANDRADE

**Course:** RO311 - Apprentissage pour la robotique @ ENSTA Paris

**Date:** September 2026

This project aims to classify emotions from speech using mel-scale spectrograms. We use the [EmoDB](http://emodb.bilderbar.info/) dataset and explore both a custom CNN and a VGG-16 transfer-learning approach, trained in Google Colab.

## Live Demo

You can try the trained models with you own recordings [here](https://diasjulia.github.io/ROB311/).

Record a few seconds of speech directly in the browser. The app computes the mel spectrogram, displays it, and runs the selected trained network (either the custom CNN or the VGG-16 based network) to show the predicted emotion.

## Methods & Results

**Dataset.** We use EmoDB (Berlin Database of Emotional Speech), which provides speaker-independent train/test splits. The dataset contains 535 utterances from 10 speakers across 7 emotion classes.

**Preprocessing.** Each audio file is sampled to 16 kHz, which were converted to mel-scale spectrograms. To standardize input size for the CNNs, we segmented each file into overlapping 1-second windows with a 0.5-second hop, treating each window as an independent training sample with the utterance's label. Spectrograms are z-score normalized using the mean and standard deviation computed on the training split only.

**Train/validation split.** Since EmoDB has only 10 speakers, we split the training data into train/validation by speaker to avoid the same speaker appearing in both sets, which would leak information across the split.

**Model.** We evaluate two architectures. The first is a CNN trained from scratch, with four convolutional blocks, followed by global average pooling and a dense classification layer. The second uses transfer learning from VGG-16 pretrained on ImageNet: the single-channel spectrogram is converted to 3 channels and rescaled to match ImageNet's input statistics, passed through fixed VGG-16 convolutional layers, flattened, and fed to a dense layer. Training for the latter proceeds in two phases, the first one with the VGG-16 base fixed, and the second with the final convolutional block unfrozen and fine-tuned. Architectural details can be found in the corresponding networks' notebooks. 

**Evaluation.** We report accuracy at two levels: window-level (each window scored independently) and file-level (softmax probabilities from all windows of a file are averaged, and the class with the highest average probability is taken as the prediction). File-level accuracy is the primary metric since it matches how predictions are used in the deployed application, where a whole recording is classified. Both networks scored accuracies at around 60%, and further details can be found in the notebooks available, including confusion matrices and other metrics (such as precision and recall) per class. 

## Repository Structure

```
├── models/                            # trained models saved as .onnx files
│   ├── cnn_scratch.onnx
│   └── transfer_vgg16.onnx
├── notebooks/                         # notebooks with training and results for each network
│   ├── RO311_CNN_from_scratch.ipynb   
│   └── RO311_transfer_learning.ipynb 
├── config.json                        # live app files
├── index.html                  
├── script.js
└── README.md
```
