from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model import classify_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'Файл отсутсвует'
    file = request.files['file']
    if file.filename == '':
        return 'Не выбран файл'
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = classify_image(filepath)
        return jsonify({'result': result})
 

if __name__ == '__main__':
    app.run(debug=True)
