# Fashion MNIST GAN

This repository contains the implementation of a Generative Adversarial Network (GAN) using TensorFlow and Keras to generate images similar to the Fashion MNIST dataset.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [References](#references)

## Overview
This project demonstrates the creation of a GAN to generate fashion images. The GAN consists of two models: a generator and a discriminator. The generator creates new images from random noise, and the discriminator attempts to distinguish between real images from the Fashion MNIST dataset and fake images produced by the generator.

## Requirements
- Python 3.9+
- TensorFlow 2.0+
- NumPy
- Matplotlib

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/chedaoo/fashion-mnist-gan.git
    cd fashion-mnist-gan
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    run the fist 2 cells of the code
    ```

## Usage
To run the code and generate fashion images, follow these steps:

1. **Import Libraries and Setup**:
    Import necessary libraries, configure TensorFlow to use GPU if available, and load the Fashion MNIST dataset.

2. **Build the Generator**:
    - The generator model transforms random noise into a 28x28 grayscale image.
    - It uses dense layers, LeakyReLU activation, reshaping, upsampling, and convolutional layers.

3. **Build the Discriminator**:
    - The discriminator model evaluates whether an input image is real or generated.
    - It consists of convolutional layers with LeakyReLU activation, dropout for regularization, and a dense output layer with a sigmoid activation.

4. **Generate and Visualize Images**:
    - Generate images using the generator and visualize them with Matplotlib.

5. **Test the Discriminator**:
    - Use the discriminator to evaluate the generated images.

## Model Architecture

### Generator
The generator takes a 128-dimensional noise vector and transforms it through several layers to produce a 28x28x1 image. Key components include:
- Dense layer to reshape the noise vector.
- Upsampling layers to increase the spatial dimensions.
- Convolutional layers to refine the image.

### Discriminator
The discriminator is a convolutional neural network that takes a 28x28x1 image as input and outputs a single scalar value indicating whether the image is real or fake. Key components include:
- Convolutional layers to extract features.
- LeakyReLU activations to introduce non-linearity.
- Dropout layers for regularization.
- A dense layer with sigmoid activation to produce the final output.

## Results
The generator creates images that resemble the fashion items from the Fashion MNIST dataset. The discriminator attempts to distinguish these generated images from real images.

## References
- [TensorFlow Documentation](https://www.tensorflow.org/guide/keras/overview)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Original GAN Paper](https://arxiv.org/abs/1406.2661)

