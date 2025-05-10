import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import model_from_json
from PIL import Image, UnidentifiedImageError

# Configuració de la pàgina
st.set_page_config(page_title="Classificador de Dígits (1-9)", layout="centered")
st.title("🔢 Classificador de Dígits (1-9)")
st.markdown("Puja una imatge d'un dígit per saber quin número és, entre el 1 i el 9!")

# Carrega la imatge pujada
uploaded_file = st.file_uploader("📤 Pujar imatge (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# Comprova que els fitxers del model existeixin
if not os.path.exists("model_digitos.json") or not os.path.exists("model_digitos.weights.h5"):
    st.error("❌ El model no s'ha trobat. Comprova que els fitxers JSON i WEIGHTS estiguin pujats correctament al repositori.")
else:
    # Carrega l'estructura del model des del fitxer JSON
    with open("model_digitos.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model_digitos.weights.h5")

    if uploaded_file:
        try:
            # Obrim la imatge, la convertim a escala de gris i la redimensionem a 28x28
            image = Image.open(uploaded_file).convert("L")
            image = image.resize((28, 28))
            st.image(image, caption='📷 Imatge pujaded', use_container_width=True)

            # Preprocessat: convertim la imatge a array, normalitzem i afegim dimensions per compatibilitat
            img_array = np.array(image).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)   # forma: (1, 28, 28)
            img_array = np.expand_dims(img_array, axis=-1)  # forma: (1, 28, 28, 1)

            # Realitzem la predicció
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            # Mapeig: com que hem filtrat el dígit 0, l'índex 0 correspon al 1, 1 al 2, etc.
            digit = predicted_class + 1
            confidence = predictions[0][predicted_class] * 100

            st.success(f"El dígit és **{digit}** amb un {confidence:.2f}% de confiança!")
        except UnidentifiedImageError:
            st.error("❌ No s'ha pogut llegir la imatge. Si us plau, puja un arxiu .jpg o .png vàlid.")
