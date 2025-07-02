import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import logging
import pickle

# Configurar logging para suprimir warnings innecesarios
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Rayos X - DenseNet121",
    page_icon="🏥",
    layout="wide"
)

# Configurar TensorFlow para usar el modo eager
tf.config.run_functions_eagerly(True)

# Título y descripción
st.title("🏥 Analizador de Rayos X para Detección de Enfermedades")
st.markdown("""
Esta aplicación utiliza un modelo DenseNet121 entrenado para detectar múltiples enfermedades en imágenes de rayos X de tórax.
Por favor, sube una imagen de rayos X en formato JPG o PNG.

**Enfermedades que puede detectar:**
- Infiltration, Effusion, Atelectasis, Nodule, Mass
""")

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesa una imagen para el modelo DenseNet121 - EXACTAMENTE igual que en el notebook
    """
    try:
        # Convertir la imagen PIL a array numpy
        img_array = np.array(image)
        
        # Convertir a escala de grises si es necesario
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                # Convertir RGBA a RGB primero
                img_array = img_array[:, :, :3]
            # Convertir RGB a escala de grises
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif len(img_array.shape) == 2:
            img = img_array
        else:
            raise ValueError("Formato de imagen no soportado")

        # Verificar que la imagen no esté vacía
        if img.size == 0:
            raise ValueError("Imagen vacía")

        # Asegurar que la imagen esté en el rango correcto
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # Verificar dimensiones mínimas
        if img.shape[0] < 32 or img.shape[1] < 32:
            raise ValueError(f"Imagen demasiado pequeña: {img.shape}")

        # Aplicar CLAHE para mejorar el contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)

        # Redimensionar
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # Convertir a RGB (DenseNet espera 3 canales)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Normalizar a float32 en rango [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # Aplicar normalización de ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_final = (img_normalized - mean) / std

        # Verificar que no hay valores NaN o infinitos
        if np.any(np.isnan(img_final)) or np.any(np.isinf(img_final)):
            raise ValueError("Valores inválidos en la imagen procesada")

        # Agregar dimensión de batch
        img_final = np.expand_dims(img_final, axis=0)
        
        return img_final

    except Exception as e:
        st.error(f"Error en preprocesamiento: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """
    Carga el modelo y las clases con cache de Streamlit
    """
    try:
        model_path = 'modelo_rayos_x.keras'
        mlb_path = 'mlb_classes.pkl'
        
        # Verificar que los archivos existen
        if not os.path.exists(model_path):
            st.error(f"""
            ❌ No se encontró el modelo entrenado: {model_path}
            
            Por favor asegúrate de:
            1. Haber entrenado el modelo ejecutando el notebook
            2. Tener el archivo modelo_rayos_x.keras en el directorio actual
            """)
            return None
            
        if not os.path.exists(mlb_path):
            st.error(f"""
            ❌ No se encontró el archivo de clases: {mlb_path}
            
            Este archivo se genera durante el entrenamiento del modelo.
            """)
            return None

        with st.spinner("Cargando modelo..."):
            # Cargar el modelo
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Cargar el MultiLabelBinarizer
            with open(mlb_path, 'rb') as f:
                mlb = pickle.load(f)
            
            class_names = mlb.classes_
            
            # Recompilar el modelo
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.F1Score(name='f1_score', threshold=0.5)
                ]
            )
            
            # Validar la forma de entrada
            expected_shape = (None, 224, 224, 3)
            actual_shape = model.input_shape
            
            if actual_shape[1:] != expected_shape[1:]:
                st.warning(f"""
                ⚠️ Advertencia: La forma de entrada del modelo {actual_shape} 
                no coincide exactamente con la esperada {expected_shape}
                """)
            
            st.success(f"✅ Modelo cargado exitosamente con {len(class_names)} clases")
            
            return model, mlb, class_names
    
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

def predict_disease(model_data, image, threshold=0.5):
    """
    Realiza predicciones de enfermedades usando el modelo
    """
    if model_data is None:
        return "❌ Modelo no disponible"
        
    model, mlb, class_names = model_data
    
    try:
        # Hacer la predicción
        predictions = model.predict(image, verbose=0)[0]
        
        # Obtener las enfermedades detectadas
        detected_diseases = []
        all_predictions = []
        
        for pred, class_name in zip(predictions, class_names):
            all_predictions.append((class_name, pred))
            if pred >= threshold:
                detected_diseases.append((class_name, pred))
        
        # Ordenar por confianza
        detected_diseases.sort(key=lambda x: x[1], reverse=True)
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Crear el resultado detallado
        result = "## 📊 Resultados del Análisis\n\n"
        
        if detected_diseases:
            result += f"### ✅ Enfermedades Detectadas (umbral ≥ {threshold:.1%}):\n"
            for disease, confidence in detected_diseases:
                confidence_emoji = "🔴" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🟠"
                result += f"{confidence_emoji} **{disease}**: {confidence:.1%}\n"
            
            # Clasificar por nivel de confianza
            high_conf = [d for d, c in detected_diseases if c >= 0.8]
            medium_conf = [d for d, c in detected_diseases if 0.6 <= c < 0.8]
            low_conf = [d for d, c in detected_diseases if c < 0.6]
            
            result += "\n### 📈 Interpretación:\n"
            if high_conf:
                result += f"🔴 **Alta confianza**: {', '.join(high_conf)}\n"
            if medium_conf:
                result += f"🟡 **Confianza media**: {', '.join(medium_conf)}\n"
            if low_conf:
                result += f"🟠 **Baja confianza**: {', '.join(low_conf)}\n"
                
        else:
            result += f"### ✅ No se detectaron enfermedades con confianza ≥ {threshold:.1%}\n"
            result += "La imagen parece normal según el análisis del modelo.\n"
        
        # Mostrar top 5 predicciones independientemente del umbral
        result += "\n### 📋 Top 5 Predicciones (todas las probabilidades):\n"
        for i, (disease, confidence) in enumerate(all_predictions[:5], 1):
            result += f"{i}. {disease}: {confidence:.1%}\n"
        
        # Añadir disclaimers importantes
        result += "\n---\n"
        result += "### ⚠️ **IMPORTANTE - Limitaciones y Disclaimers:**\n"
        result += "- 🩺 **Este análisis NO reemplaza el diagnóstico médico profesional**\n"
        result += "- 📊 Las predicciones se basan en patrones aprendidos de datos históricos\n"
        result += "- 🎯 La precisión puede variar según la calidad de la imagen\n"
        result += "- 👨‍⚕️ **Siempre consulte con un radiólogo o médico calificado**\n"
        result += "- 🔄 Considere obtener una segunda opinión para casos complejos\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error durante la predicción: {str(e)}"

def display_model_info(model_data):
    """
    Muestra información sobre el modelo cargado
    """
    if model_data is None:
        return
        
    model, mlb, class_names = model_data
    
    with st.expander("ℹ️ Información del Modelo"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Arquitectura:** DenseNet121 pre-entrenado")
            st.write(f"**Parámetros totales:** {model.count_params():,}")
            st.write(f"**Forma de entrada:** {model.input_shape}")
            st.write(f"**Número de clases:** {len(class_names)}")
        
        with col2:
            st.write("**Clases detectables:**")
            for i, class_name in enumerate(class_names, 1):
                st.write(f"{i}. {class_name}")

# Interfaz principal
def main():
    # Cargar el modelo
    model_data = load_model()
    
    if model_data is not None:
        display_model_info(model_data)
    
    # Widget para subir archivo
    uploaded_file = st.file_uploader(
        "📁 Sube una imagen de rayos X", 
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, JPEG, PNG. Tamaño recomendado: al menos 224x224 píxeles"
    )
    
    if uploaded_file is not None:
        # Mostrar la imagen original
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Imagen Original")
            st.image(image, caption=f"Archivo: {uploaded_file.name}", use_container_width=True)
            
            # Información de la imagen
            st.write(f"**Dimensiones:** {image.size[0]} x {image.size[1]} píxeles")
            st.write(f"**Modo:** {image.mode}")
            st.write(f"**Tamaño del archivo:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
        
        with col2:
            st.subheader("⚙️ Configuración del Análisis")
            
            # Slider para ajustar el umbral de detección
            threshold = st.slider(
                "🎯 Umbral de detección",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Ajusta el umbral de confianza. Valores más altos = detecciones más precisas pero menos sensibles."
            )
            
            # Mostrar interpretación del umbral
            if threshold >= 0.8:
                st.info("🔴 Umbral alto: Solo detecciones muy confiables")
            elif threshold >= 0.6:
                st.info("🟡 Umbral medio: Balance entre precisión y sensibilidad")
            else:
                st.info("🟠 Umbral bajo: Más sensible, puede incluir falsos positivos")
        
        # Botón para procesar la imagen
        if st.button("🔍 Analizar Imagen", type="primary", use_container_width=True):
            if model_data is None:
                st.error("❌ No se puede analizar: modelo no disponible")
                return
            
            with st.spinner("🔄 Analizando la imagen... Esto puede tomar unos momentos."):
                # Preprocesar la imagen
                processed_image = preprocess_image(image)
                
                if processed_image is not None:
                    # Hacer la predicción
                    prediction = predict_disease(model_data, processed_image, threshold)
                    st.markdown(prediction)
                else:
                    st.error("❌ Error al procesar la imagen. Por favor, intenta con otra imagen.")

# Sidebar con información adicional
def sidebar_info():
    st.sidebar.title("📋 Guía de Uso")
    st.sidebar.markdown("""
    ### 🚀 Pasos para usar la aplicación:
    
    1. **📁 Sube una imagen** de rayos X de tórax
    2. **⚙️ Ajusta el umbral** según tus necesidades
    3. **🔍 Presiona "Analizar"** y espera los resultados
    4. **📊 Revisa las predicciones** y probabilidades
    
    ### 📏 Recomendaciones de imagen:
    - **Formato:** JPG, JPEG, PNG
    - **Tamaño:** Mínimo 224x224 píxeles
    - **Calidad:** Imagen clara y bien contrastada
    - **Orientación:** Rayos X de tórax frontal (PA)
    
    ### 🎯 Interpretación de umbrales:
    - **0.1-0.4:** Muy sensible (muchos resultados)
    - **0.5-0.7:** Equilibrado (recomendado)
    - **0.8-0.9:** Muy específico (solo alta confianza)
    """)
    
    st.sidebar.title("⚠️ Advertencias Médicas")
    st.sidebar.warning("""
    **IMPORTANTE:**
    
    - Esta herramienta es solo de apoyo
    - NO reemplaza el diagnóstico médico
    - Siempre consulte con un profesional
    - Los resultados pueden contener errores
    - Use solo como referencia preliminar
    """)
    
    st.sidebar.title("🔧 Información Técnica")
    st.sidebar.info("""
    **Modelo:** DenseNet121 pre-entrenado
    
    **Dataset:** NIH Chest X-ray Dataset
    
    **Precisión:** Variable según la enfermedad
    
    **Última actualización:** Modelo mejorado
    """)

if __name__ == "__main__":
    sidebar_info()
    main() 