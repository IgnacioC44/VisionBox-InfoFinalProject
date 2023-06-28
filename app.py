from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image

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

    # Carga el modelo pre-entrenado (ResNet50)
    model = tf.keras.applications.ResNet50()
    
    # Preprocesamiento de la imagen
    img = Image.open(image_path).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    # Realiza la predicción
    predictions = model.predict(img_array)
    results = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)[0]
    
    # Extrae los objetos reconocidos y su precisión
    objects = []
    for result in results:
        objects.append({'name': result[1], 'precision': round(result[2] * 100, 2)})
    
    # Renderiza la página de resultados y muestra los objetos reconocidos
    return render_template('results.html', objects=objects, image_path=image_path)


if __name__ == '__main__':
    app.run()