import os
import sqlite3

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, g
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

DATABASE = 'predictions.db'
app.config['UPLOAD_FOLDER'] = 'uploads'


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # This allows access to the rows as dictionaries
    return db


def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE,
            prediction TEXT
        )
        ''')
        db.commit()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def insert_db(query, args=()):
    db = get_db()
    db.execute(query, args)
    db.commit()


def delete_db(query, args=()):
    db = get_db()
    db.execute(query, args)
    db.commit()


@app.route('/')
def index():
    predictions = query_db('SELECT * FROM predictions')
    return render_template('index.html', predictions=predictions)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify("No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify("No selected file"), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        image = image.resize((64, 64))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0

        model = load_model('saved_model_and_weights/model_cnn3.h5')
        prediction = model.predict(image)
        probability = prediction[0][0]
        result = "Positive" if probability > 0.5 else "Negative"

        insert_db('INSERT INTO predictions (image_path, prediction) VALUES (?, ?)', [filepath, result])
        return jsonify(image_path=filepath, prediction=result)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/delete/<int:id>', methods=['POST'])
def delete_prediction(id):
    prediction = query_db('SELECT * FROM predictions WHERE id = ?', [id], one=True)
    if prediction:
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], prediction['image_path']))
        except OSError:
            pass  # Handle the error.
        delete_db('DELETE FROM predictions WHERE id = ?', [id])
    return jsonify({"success": "Record deleted"})


@app.route('/predictions', methods=['GET'])
def get_predictions():
    predictions = query_db('SELECT * FROM predictions')
    return jsonify([{'id': x['id'], 'image_path': x['image_path'], 'prediction': x['prediction']} for x in predictions])


if __name__ == '__main__':
    init_db()  # Ensure the database and table are created
    app.run(debug=True)
