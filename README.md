# Deep Learning Practicals (1–10)

A collection of **10 Python/Jupyter Notebook practicals** covering foundational deep learning, image classification, sequence models, generative models, and web deployment. The practicals follow the experiments in **`DL-Practicals.docx`**, with minor code and environment adjustments where needed to make the exercises runnable.

The notebooks work primarily in **Google Colab** and can also be run in **VS Code with a Python Jupyter kernel**, provided the required packages are installed. The lab is intended for learning and experimentation, not production-ready model deployment.

## Practicals

| No. | Notebook | Experiment | Key concepts |
| --- | --- | --- | --- |
| 1 | [`Practical_01_NumPy_Pandas_TensorFlow_PyTorch.ipynb`](Practical_01_NumPy_Pandas_TensorFlow_PyTorch.ipynb) | Exploring and plotting data | NumPy, Pandas, Matplotlib, TensorFlow, PyTorch |
| 2 | [`Practical_02_Single_Layer_Perceptron.ipynb`](Practical_02_Single_Layer_Perceptron.ipynb) | Single-layer perceptron for binary classification | AND gate, NumPy, TensorFlow |
| 3 | [`Practical_03_MLP_MNIST.ipynb`](Practical_03_MLP_MNIST.ipynb) | MNIST digit classification using an MLP | Keras, dense layers, classification |
| 4 | [`Practical_04_CNN_CIFAR10.ipynb`](Practical_04_CNN_CIFAR10.ipynb) | CNN for CIFAR-10 image classification | Convolution, pooling, softmax |
| 5 | [`Practical_05_Transfer_Learning_Updated.ipynb`](Practical_05_Transfer_Learning_Updated.ipynb) | Custom image classification with VGG16/ResNet50 | Transfer learning, fine-tuning, cats vs. dogs |
| 6 | [`Practical_06_RNN_Sentiment_Analysis.ipynb`](Practical_06_RNN_Sentiment_Analysis.ipynb) | IMDB movie review sentiment analysis | RNN, embeddings, optional LSTM |
| 7 | [`Practical_07_LSTM_Stock_Prediction.ipynb`](Practical_07_LSTM_Stock_Prediction.ipynb) | Predicting stock prices from historical data | LSTM, time series, scaling, RMSE |
| 8 | [`Practical_08_Autoencoders.ipynb`](Practical_08_Autoencoders.ipynb) | Compressing and reconstructing MNIST images | Encoder, latent space, decoder |
| 9 | [`Practical_09_GAN_MNIST.ipynb`](Practical_09_GAN_MNIST.ipynb) | Generating handwritten digits using GANs | Generator, discriminator, adversarial training |
| 10 | [`Practical_10_Model_Deployment_Updated.ipynb`](Practical_10_Model_Deployment_Updated.ipynb) | Train and deploy an MNIST model | TensorFlow/Keras, Flask, Streamlit |

The earlier Practical 5 and Practical 10 notebooks are retained for reference. **Use the `Updated` versions** for the corrected dataset/deployment workflows.

## Requirements

- Python 3.x (for local deployment, **Python 3.10** is a known environment used with TensorFlow 2.16.2; use versions supported by your installed TensorFlow release)
- Jupyter Notebook or Google Colab
- NumPy, Pandas, Matplotlib
- TensorFlow / Keras, PyTorch
- Scikit-learn and OpenCV where needed
- Flask and Streamlit for Practical 10

For local Jupyter notebooks, install the basics into the **same Python environment as the notebook kernel**:

```powershell
python -m pip install jupyter ipykernel numpy pandas matplotlib tensorflow torch torchvision scikit-learn opencv-python
```

Practical 10 has its own deployment dependencies in `Practical_10_Deployment_Updated/requirements.txt`. Avoid mixing training and deployment Python environments without checking TensorFlow/Keras compatibility.

## How to run

### Option A: Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook** and upload the `.ipynb` for the practical.
3. Run the cells **in order**. For heavier models, select **Runtime → Change runtime type** and choose a GPU if available.
4. Upload any external dataset the notebook requests (see below).
5. Save your executed notebook to retain the outputs.

### Option B: VS Code / Jupyter (Windows)

1. Install the **Python** and **Jupyter** extensions in VS Code.
2. Open the notebook and choose a **Python 3 kernel**, not PowerShell.
3. Install packages in that kernel's Python environment. In a notebook cell, `%pip install pandas numpy matplotlib` installs into the active kernel environment.
4. Run the notebook cells from top to bottom.

Training deep learning models locally can use substantial RAM and CPU time. If a notebook raises `MemoryError`, try Google Colab or reduce the training samples/batch size for a clearly marked trial run. Do not describe reduced-data results as full-dataset results.

## Datasets and special setup

**Practicals 3, 4, 6, 8, 9 and 10:** MNIST, CIFAR-10, or IMDB are loaded through the dataset helpers shown in the notebooks. Their first run may require internet access to download data. Practical 1 and Practical 2 use examples defined in their notebooks.

### Practical 5 — cats and dogs

Organize your original downloaded photos by category. The updated notebook can build training, validation, and test splits automatically. The local folder layout used during this project is:

```text
dataset/
├── downloaded_images/
│   ├── cats_set/          # original cat images
│   └── dogs_set/          # original dog images
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

The split-preparation cell copies images into **70% training / 15% validation / 15% testing** folders. Set paths relative to the notebook's working directory (`Path.cwd()`) in VS Code; `/content/` is specific to Google Colab. For custom-image predictions, the image must exist or a test image can be selected from `dataset/test/`. The actual class mapping comes from `train_generator.class_indices`.

### Practical 7 — historical stock CSV

Place your CSV in `dataset/` and point the notebook at its actual filename, such as `dataset/Book1.csv` or `dataset/stock_data.csv`. The source experiment uses a **`Close`** column for stock closing prices and sequences of historical values. Check the data columns before training. A column labeled `Closing Volume` should be used as the target **only if you confirm that it represents closing prices**, rather than trading volume. On Windows, prefer `pathlib.Path` paths to unescaped strings such as `"D:\\..."`.

### Practical 8 — memory requirements

Autoencoder training on full MNIST can require substantial temporary memory. If local training runs out of memory, use Colab or train in smaller batches. Report any reduced-dataset experiment clearly.

### Practical 9 — GAN runtime

GAN training is computationally intensive. The notebook provides a shorter trial run and a larger iteration count corresponding to the source experiment. Images are generated by the model after training; their appearance will depend on training duration and settings.

## Practical 10 — Flask and Streamlit deployment

Use the updated notebook to train an MNIST digit classifier and export:

- `mnist_model.keras` — saved Keras model
- `mnist_weights.npz` — exported model weights, useful for avoiding cross-version `.h5` model deserialization problems

The deployment source is available in **`Practical_10_Deployment_Updated/`** (and as a ZIP archive). Copy at least `mnist_weights.npz` into the deployment folder root, **beside `requirements.txt` / `model_utils.py`**, before starting either app:

```text
Practical_10_Deployment_Updated/
├── flask_app/
│   ├── app.py
│   └── templates/
├── streamlit_app/
│   └── app.py
├── model_utils.py
├── mnist_weights.npz        # export from training notebook
├── mnist_model.keras        # optional model export
└── requirements.txt
```

From the deployment folder root, install its requirements:

```powershell
python -m pip install -r requirements.txt
```

**Run Flask:**

```powershell
cd flask_app
python app.py
```

Then visit <http://127.0.0.1:5000>.

**Run Streamlit:** From the deployment root, in a separate terminal:

```powershell
cd streamlit_app
python -m streamlit run app.py
```

Then visit <http://localhost:8501> (or the address shown in the terminal).

> The practical's simple app sends a uniform pixel value as a 28 × 28 image to illustrate inference and deployment. It is **not** a handwriting canvas or a meaningful benchmark for recognition quality.

For more information, see [`Practical_10_Deployment_Updated/README.md`](Practical_10_Deployment_Updated/README.md).

## Notes

- **Source:** `DL-Practicals.docx` (user-provided practical manual). Notebook exercises follow its sequence, with code corrections and environment-specific adjustments where necessary.
- Each practical includes objectives/theory, code and expected or sample outputs as appropriate. **Questions/viva sections are intentionally excluded.**
- Expected outputs in notebook Markdown are examples from the manual; **they are not proof of successful execution**. Actual accuracy and predictions can vary.
- Downloaded datasets, trained model files, and generated outputs may be large; keep only the assets needed for your repository and avoid committing personal or private data.

---

**Completion tip:** Run each notebook fully, inspect actual outputs, and save the executed `.ipynb` before submitting the practical.
