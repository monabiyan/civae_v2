# CI-VAE: Class-Informed Variational Autoencoder


Build and Run the Docker image:

docker build -t ci-vae .

docker run --rm -v "$(pwd)/output:/app/output" ci-vae > results.zip
