# Clasificador mnist con pytorch
Clasificador de Dígitos MNIST con PyTorch
<div align="center">
Mostrar imagen
Mostrar imagen
Mostrar imagen
Proyecto modular de Machine Learning para clasificación de dígitos escritos a mano usando redes neuronales
Características •
Instalación •
Uso •
Estructura •
Resultados
</div>

 Descripción
Este proyecto implementa un clasificador de dígitos utilizando el famoso dataset MNIST. Está diseñado con una arquitectura modular que facilita el mantenimiento, la escalabilidad y el aprendizaje de buenas prácticas en Machine Learning.
 Características

Arquitectura modular y bien organizada
Red neuronal configurable con dropout
Visualización automática de resultados
Soporte para GPU y CPU
Código completamente documentado en español
Guardado automático de modelos y gráficas
Configuración centralizada de hiperparámetros

 Instalación
Requisitos previos

Python 3.8 o superior
pip (gestor de paquetes de Python)

Pasos de instalación

Clonar el repositorio

bashgit clone https://github.com/tu-usuario/clasificador-mnist-pytorch.git
cd clasificador-mnist-pytorch

Crear un entorno virtual (recomendado)

bash# En Windows
python -m venv venv
venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate

Instalar dependencias

bashpip install -r requirements.txt


Uso: Entrenamiento rápido


Para entrenar el modelo con la configuración por defecto:
bashpython principal.py


Funcionamiento del programa:


Descarga el dataset MNIST
Crea el modelo
Entrena durante 5 épocas
Guarda el modelo en modelos/modelo_mnist.pth
Genera gráficas en resultados/graficas_entrenamiento.png

Personalizar el entrenamiento
Edita configuracion.py para modificar los hiperparámetros:
python# Ejemplo de configuración personalizada
EPOCAS = 10
TASA_APRENDIZAJE = 0.0005
TAMAÑO_LOTE = 128
NEURONAS_OCULTA1 = 256
Uso programático
pythonfrom modelo import RedNeuronalSimple
from datos import cargar_mnist
from entrenamiento import entrenar_modelo
import torch

# Cargar datos
cargador_ent, cargador_pru = cargar_mnist(tamaño_lote=64)

# Crear modelo
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = RedNeuronalSimple().to(dispositivo)

# Entrenar
criterio = torch.nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

historial = entrenar_modelo(
    modelo, cargador_ent, cargador_pru,
    criterio, optimizador, dispositivo, epocas=5
)
Estructura del Proyecto
clasificador-mnist-pytorch/
│
├── principal.py              # Script principal - punto de entrada
├── modelo.py                 # Arquitectura de la red neuronal
├── datos.py                  # Carga y preparación de datos
├── entrenamiento.py          # Lógica de entrenamiento y evaluación
├── utilidades.py             # Funciones auxiliares
├──configuracion.py          # Hiperparámetros centralizados
│
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Documentación (este archivo)
├── .gitignore               # Archivos ignorados por Git
│
├── datos/                    # Dataset MNIST (auto-descargado)
├── modelos/                  # Modelos entrenados (.pth)
│   └── .gitkeep
└── resultados/               # Gráficas y métricas
    └── .gitkeep
Descripción de módulos
ArchivoDescripciónmodelo.pyDefine la clase RedNeuronalSimple con arquitectura configurabledatos.pyGestiona descarga, transformaciones y DataLoaders de MNISTentrenamiento.py
Implementa el loop de entrenamiento y evaluaciónutilidades.pyFunciones para guardar/cargar modelos y visualizar resultadosconfiguracion.py
Centraliza todos los hiperparámetros del proyectoprincipal.pyOrquesta el flujo completo del entrenamiento
Arquitectura del Modelo
La red neuronal implementada tiene la siguiente estructura:
Entrada (784) 
    ↓
Capa Oculta 1 (128 neuronas)
    ↓
ReLU + Dropout(0.2)
    ↓
Capa Oculta 2 (64 neuronas)
    ↓
ReLU + Dropout(0.2)
    ↓
Salida (10 clases)
El entrenamiento genera automáticamente gráficas de:

 Evolución de la pérdida (loss)
 Evolución de la precisión (accuracy)

<div align="center">
<i>Ejemplo de gráficas generadas durante el entrenamiento</i>
</div>
Tiempo de entrenamiento

CPU: ~2-5 minutos
GPU: ~30-60 segundos

 Configuración
Todos los hiperparámetros están centralizados en configuracion.py:
python# Datos
TAMAÑO_LOTE = 64
RUTA_DATOS = './datos'

# Arquitectura
NEURONAS_OCULTA1 = 128
NEURONAS_OCULTA2 = 64
TASA_DROPOUT = 0.2

# Entrenamiento
EPOCAS = 5
TASA_APRENDIZAJE = 0.001

# GPU/CPU
USAR_GPU = True  # Usa GPU si está disponible
 Testing
Para verificar que todo funciona correctamente:
python# Prueba rápida del modelo
python -c "from modelo import RedNeuronalSimple; m = RedNeuronalSimple(); print(' Modelo creado correctamente')"

# Prueba de carga de datos
python -c "from datos import cargar_mnist; cargar_mnist(); print(' Datos cargados correctamente')"
 Roadmap y Mejoras Futuras

 Script de inferencia para hacer predicciones
 Interfaz web con Gradio/Streamlit
 Implementar CNN para mejor precisión
 Data augmentation
 Validación cruzada (k-fold)
 Early stopping
 Integración con TensorBoard
 Exportar a ONNX para producción
 Dockerizar el proyecto

 Contribuciones
Las contribuciones son bienvenidas y apreciadas. Para contribuir:

 Fork del proyecto
 Crea una rama para tu feature (git checkout -b feature/nueva-funcionalidad)
 Commit de tus cambios (git commit -m 'Añadir nueva funcionalidad')
 Push a la rama (git push origin feature/nueva-funcionalidad)
 Abre un Pull Request

Guías de contribución

Mantén el código en español
Documenta todas las funciones
Sigue el estilo PEP 8
Añade tests si es posible

 Autor
Daniel Bernad Peñuelas

GitHub: Daniel Bernad Peñuelas
Email: bernadd2003@gmail.com

 Agradecimientos

Dataset MNIST por Yann LeCun
PyTorch por Facebook AI Research
La comunidad de Machine Learning


<div align="center">
 Si este proyecto te ha sido útil, considera darle una estrella 
</div>
