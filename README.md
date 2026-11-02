# MNIST Denoising Autoencoder

Notebook: `denoising-autoencoder.ipynb`

## Provenance
This is my own university coursework, published here with permission to publish it. The notebook was assembled from cells of a larger coursework extract; the only addition is one setup line, `X_train_tensor = torch.from_numpy(X_train).float().div(255).unsqueeze(1)`, placed after the imports.

## What it does
- Trains two small convolutional autoencoders (10 epochs, NAdam, MSE) whose input noise is an `nn.Dropout` layer at the front, comparing dropout rates **0.1 and 0.2**.
- Visualizes reconstructions of **five training examples** in eval mode (dropout off) and prints MSE on those five images.
- This is **not** a held-out denoising evaluation: no test split is used, and no performance claims are made.

## Cell dependencies
Cell 1 loads MNIST and defines `X_train`; cell 2 builds `X_train_tensor` from it; the model, training, and plotting cells depend on those in order. Run cells top to bottom.

## Dependencies
- Dataset: MNIST via `torchvision.datasets.MNIST`, downloaded into `./data` on first run.
- Python packages: see `requirements.txt` (unpinned).

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook denoising-autoencoder.ipynb
```

## Status
- **No fresh end-to-end run has been done, and reproducibility has not been verified.**
- The repository contains no data, outputs, weights, or secrets.
