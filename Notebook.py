# 1. IMPORTAR LIBRERÍAS NECESARIAS
import tensorflow as tf                         # Importa TensorFlow para construir y entrenar modelos de aprendizaje profundo
from tensorflow.keras import layers, models     # Importa módulos de Keras para definir capas y modelos secuenciales
import matplotlib.pyplot as plt                 # Importa Matplotlib para visualizar gráficos
import numpy as np                              # Importa NumPy para operaciones numéricas y manejo de arrays

# 2. CARGA Y FILTRADO DEL DATASET MNIST
# Se carga el dataset MNIST, que incluye dígitos del 0 al 9. 
# En este ejemplo, se filtra para conservar solo ejemplos en los que el dígito no es 0.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # Carga de datos de entrenamiento y prueba

# Se obtienen los índices en los que la etiqueta (dígito) no es 0
train_filter = np.where(y_train != 0)  # Índices de entrenamiento donde el dígito no es 0
test_filter  = np.where(y_test != 0)   # Índices de prueba donde el dígito no es 0

# Se aplican los filtros y se reindexan las etiquetas restando 1 para pasar de 1-9 a 0-8
x_train, y_train = x_train[train_filter], y_train[train_filter] - 1  # Filtrado y ajuste de datos de entrenamiento
x_test, y_test   = x_test[test_filter], y_test[test_filter] - 1       # Filtrado y ajuste de datos de prueba

# 3. PREPROCESADO DE LAS DATOS
# Se redimensionan las imágenes para incluir la dimensión del canal (imágenes en escala de grises)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Redimensiona, convierte a float32 y normaliza las imágenes de entrenamiento
x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0    # Redimensiona, convierte a float32 y normaliza las imágenes de prueba

# 4. DEFINICIÓN DE UN MODELO CONVOLUCIONAL LIGERO
model = models.Sequential([                         # Crea un modelo secuencial
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Primera capa convolucional: 32 filtros de 3x3, ReLU y definición de la forma de entrada
    layers.MaxPooling2D((2, 2)),                    # Primera capa de max pooling: reduce la dimensión espacial
    layers.Conv2D(64, (3, 3), activation='relu'),   # Segunda capa convolucional: 64 filtros de 3x3 y ReLU
    layers.MaxPooling2D((2, 2)),                    # Segunda capa de max pooling: reduce aún más la dimensión espacial
    layers.Flatten(),                             # Aplanamiento de la salida para convertirla en un vector
    layers.Dense(64, activation='relu'),          # Capa densa oculta con 64 neuronas y función de activación ReLU
    # Capa de salida: contiene 9 neuronas correspondientes a los dígitos 1-9 (reindexados a 0-8),
    # y usa la función 'softmax' para obtener una distribución de probabilidades sobre las clases.
    layers.Dense(9, activation='softmax')         
])

# Compilación del modelo mediante la configuración del optimizador y la función de pérdida
model.compile(optimizer='adam',                       # Utiliza el optimizador Adam, eficiente para entrenamiento
              loss='sparse_categorical_crossentropy',   # Función de pérdida adecuada para clasificación multiclase con etiquetas enteras
              metrics=['accuracy'])                     # Métrica para evaluar la precisión del modelo

# 5. ENTRENAMIENTO DEL MODELO
history = model.fit(                  # Entrena el modelo con los datos de entrenamiento
    x_train, y_train,                # Imágenes y etiquetas de entrenamiento
    epochs=5,                        # Número de épocas (iteraciones sobre el dataset completo)
    validation_data=(x_test, y_test)   # Datos de validación para evaluar el rendimiento en cada época
)

# 6. EXPORTACIÓN DEL MODELO Y SUS PESOS
# Exporta la arquitectura del modelo en formato JSON
model_json = model.to_json()                     # Convierte la estructura del modelo a un formato JSON
with open("model_digitos.json", "w") as json_file: # Abre (o crea) un archivo para escribir la arquitectura del modelo
    json_file.write(model_json)                  # Guarda la estructura en el archivo

# Guarda los pesos del modelo en un archivo con extensión H5
model.save_weights("model_digitos.weights.h5")    # Exporta los pesos entrenados del modelo

# 7. VISUALIZACIÓN DE LA EVOLUCIÓN DE LA PRECISIÓN
plt.figure(figsize=(8, 6))                          # Crea una figura de 8x6 pulgadas para el gráfico
plt.plot(history.history['accuracy'], 'r', label='Entrenamiento')  # Grafica la precisión del entrenamiento en color rojo
plt.plot(history.history['val_accuracy'], 'b', label='Validación')   # Grafica la precisión de validación en color azul
plt.title('Evolución de la Precisión')              # Título del gráfico
plt.xlabel('Época')                                # Etiqueta del eje X
plt.ylabel('Precisión')                            # Etiqueta del eje Y
plt.legend()                                       # Muestra la leyenda para identificar las curvas
plt.show()                                         # Muestra el gráfico resultante
