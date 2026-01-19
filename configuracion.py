"""
Archivo de configuración con los hiperparámetros del modelo
"""

# Configuración de datos
RUTA_DATOS = './datos'
TAMAÑO_LOTE = 64

# Configuración del modelo
NEURONAS_ENTRADA = 784  # 28x28 píxeles
NEURONAS_OCULTA1 = 128
NEURONAS_OCULTA2 = 64
NEURONAS_SALIDA = 10  # 10 dígitos (0-9)
TASA_DROPOUT = 0.2

# Configuración del entrenamiento
EPICAS = 5
TASA_APRENDIZAJE = 0.001

# Rutas de guardado
RUTA_MODELO = 'modelos/modelo_mnist.pth'
RUTA_GRAFICAS = 'resultados/graficas_entrenamiento.png'

# Configuración del dispositivo
# Se seleccionará automáticamente GPU si está disponible, sino CPU
USAR_GPU = True