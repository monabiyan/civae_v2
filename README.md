# CI-VAE: Class-Informed Variational Autoencoder

This repository provides a minimal implementation of a class informed
variational autoencoder (CI-VAE) along with example scripts for training
on MNIST and performing inference.

## Training example

To train a CI-VAE on MNIST and generate several plots and GIFs run:

```bash
python example_mnist.py
```

All outputs are written to the `output/` directory.

## Inference example

After training you can create a video showing a smooth transition between
two randomly chosen digit *3* images from the MNIST test set. First save
the trained model as a checkpoint file with the keys `input_size`,
`n_classes`, `latent_size` and `state_dict`. Then run:

```bash
python mnist_interpolate_inference.py --model path/to/checkpoint.pt
```

The script produces `digit3_transition.gif` in the specified output
folder which visualises the interpolation between the two images.

### Docker

To build and run the Docker image use:

```bash
docker build -t ci-vae .
docker run --rm -v "$(pwd)/output:/app/output" ci-vae > results.zip
```
