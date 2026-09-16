# Clasificador de Dígitos MNIST con PyTorch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Este proyecto implementa un sistema de reconocimiento de dígitos manuscritos (0 al 9) utilizando el dataset **MNIST** y el framework **PyTorch**. Está diseñado bajo una arquitectura totalmente modular que promueve las buenas prácticas en Machine Learning, facilitando su mantenimiento, escalabilidad y experimentación.

---

## Características Principales

-  **Arquitectura Modular**: Código estructurado y desacoplado por responsabilidades (datos, modelo, entrenamiento, utilidades).
-  **Configuración Centralizada**: Gestión sencilla de hiperparámetros desde un único archivo.
-  **Red Neuronal Configurable**: Soporte para capas ocultas personalizables y regularización con *Dropout*.
-  **Soporte GPU / CPU**: Detección y aprovechamiento automático de CUDA.
-  **Visualización Automática**: Generación y guardado de gráficas de rendimiento (*loss* y *accuracy*).
-  **Persistencia**: Guardado automático del modelo entrenado.
-  **Código Documentado**: Comentarios explicativos y tipado en español.

---

## Estructura del Proyecto

```text
clasificador-mnist-pytorch/
├── datos/                  # Carpeta para descarga y almacenamiento del dataset
├── modelos/                # Almacenamiento de modelos entrenados (.pth)
├── resultados/             # Gráficas e imágenes generadas
├── configuracion.py        # Hiperparámetros y ajustes globales
├── datos.py                # Carga, transformación y DataLoaders de MNIST
├── modelo.py               # Definición de la arquitectura de la Red Neuronal
├── entrenamiento.py        # Bucle de entrenamiento y evaluación
├── utilidades.py           # Funciones auxiliares (guardado, gráficas)
├── principal.py            # Script principal de ejecución
└── requirements.txt        # Dependencias del proyecto
