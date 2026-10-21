from flask import Flask, render_template, request
import numpy as np
from model_utils import load_mnist_model

app = Flask(__name__)
model = load_mnist_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    value = 100
    if request.method == "POST":
        try:
            value = int(request.form.get("pixel_value", 100))
            if not 0 <= value <= 255:
                raise ValueError("Pixel value must be between 0 and 255.")
            # The uploaded Word practical uses a constant-intensity 28x28 image.
            sample = np.full((1, 28, 28), value / 255.0, dtype=np.float32)
            probabilities = model.predict(sample, verbose=0)
            prediction = int(np.argmax(probabilities, axis=1)[0])
        except (TypeError, ValueError) as exc:
            return render_template("index.html", prediction=None, value=value, error=str(exc))
    return render_template("index.html", prediction=prediction, value=value, error=None)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
