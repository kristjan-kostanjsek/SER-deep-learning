# SER-deep-learning

This project was developed as part of the Deep Learning course at FRI and explores Speech Emotion Recognition (SER) using a computer vision-based approach. Audio signals are converted into Mel spectrograms and treated as images, enabling the use of established CNN and vision transformer architectures such as ResNet, ConvNeXt, and DINOv2.

Due to the limited size of the CREMA-D dataset, the focus was on smaller models and techniques to mitigate overfitting, including data augmentation, regularization, transfer learning, multi-task learning (emotion + intensity), and intermediate feature extraction. The best CNN-based model achieved an accuracy of 67.69%.

For comparison, a wav2vec 2.0 baseline was also evaluated, reaching 71.46% accuracy with minimal tuning, highlighting the strength of self-supervised audio-based approaches.

Full report (PDF): [Deep Computer Vision Models for Speech Emotion Recognition](./doc/report.pdf)
