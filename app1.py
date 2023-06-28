from flask import Flask, render_template, request
from imageai.Detection import ObjectDetection
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Obtén el archivo de imagen enviado por el usuario
    file = request.files['file']
    
    # Guarda la imagen en disco
    image_path = 'static/' + file.filename
    file.save(image_path)

    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , image_path), output_image_path=os.path.join(execution_path , image_path), minimum_percentage_probability=30)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")
    # Renderiza la página de resultados y muestra los objetos reconocidos con la imagen señalada
    return render_template('results.html', image_path = image_path)

if __name__ == '__main__':
    app.run()
