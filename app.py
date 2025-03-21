from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image

app = Flask(__name__)
model = load_model('model/fibrosis_model.h5')

LABELS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]  # Fibrosis stages

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = "static/uploads/" + file.filename
        file.save(file_path)

        img = preprocess_image(file_path)
        prediction = model.predict(img)
        stage = LABELS[np.argmax(prediction)]

        return render_template('index.html', filename=file.filename, stage=stage)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)