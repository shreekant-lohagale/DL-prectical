# Practical 10 — Flask / Streamlit deployment

Source experiment: Practical 10 of `DL-Practicals.docx`. This package also adds cross-version model-loading support, which is an implementation fix rather than a change to the experiment.

## Train and export
1. Open `Practical_10_Model_Deployment_Updated.ipynb` in VS Code or Google Colab.
2. Run through Part A and save `mnist_model.keras` and `mnist_weights.npz`.
3. Place `mnist_weights.npz` (recommended) in this package's **root** folder (alongside `model_utils.py`). You may also copy `mnist_model.keras` there. The app tries the numeric weights first.

## Install packages
Open a VS Code PowerShell terminal in this package's root and use the same Python environment for installation and application execution:

```powershell
python -m pip install -r requirements.txt
```

## Run Flask
```powershell
cd flask_app
python app.py
```
Open http://127.0.0.1:5000 .

## Run Streamlit
Return to the package root, then:
```powershell
cd streamlit_app
python -m streamlit run app.py
```
Open the local URL displayed (usually http://localhost:8501).

The Word practical demonstrates predictions from a **single pixel-value slider** by forming a constant 28x28 image, not by uploading handwritten digits. Such predictions are not necessarily meaningful. No pretrained model is included in this package; you must run the notebook training cells first.
