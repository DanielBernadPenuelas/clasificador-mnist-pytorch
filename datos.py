from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def obtener_transformaciones():
    """
    Define las transformaciones a aplicar a las imágenes.
    
    Returns:
        Composición de transformaciones de torchvision
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def cargar_mnist(ruta_datos='./datos', tamaño_lote=64):
    """
    Carga el dataset MNIST y crea los cargadores de datos.
    
    Args:
        ruta_datos: Ruta donde guardar/cargar el dataset (default: './datos')
        tamaño_lote: Tamaño del lote para el entrenamiento (default: 64)
        
    Returns:
        tuple: (cargador_entrenamiento, cargador_prueba)
    """
    transformacion = obtener_transformaciones()
    
    # Descargar y cargar conjunto de entrenamiento
    conjunto_entrenamiento = datasets.MNIST(
        root=ruta_datos,
        train=True,
        download=True,
        transform=transformacion
    )
    
    # Descargar y cargar conjunto de prueba
    conjunto_prueba = datasets.MNIST(
        root=ruta_datos,
        train=False,
        download=True,
        transform=transformacion
    )
    
    # Crear cargadores de datos
    cargador_entrenamiento = DataLoader(
        conjunto_entrenamiento,
        batch_size=tamaño_lote,
        shuffle=True
    )
    
    cargador_prueba = DataLoader(
        conjunto_prueba,
        batch_size=tamaño_lote,
        shuffle=False
    )
    
    return cargador_entrenamiento, cargador_prueba


def obtener_info_dataset(cargador_entrenamiento, cargador_prueba):
    """
    Muestra información sobre el dataset cargado.
    
    Args:
        cargador_entrenamiento: DataLoader de entrenamiento
        cargador_prueba: DataLoader de prueba
    """
    num_entrenamiento = len(cargador_entrenamiento.dataset)
    num_prueba = len(cargador_prueba.dataset)
    num_lotes = len(cargador_entrenamiento)
    
    print(f"Muestras de entrenamiento: {num_entrenamiento}")
    print(f"Muestras de prueba: {num_prueba}")
    print(f"Número de lotes: {num_lotes}")
    print(f"Tamaño del lote: {cargador_entrenamiento.batch_size}")