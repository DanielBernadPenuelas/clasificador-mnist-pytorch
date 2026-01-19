import torch
import torch.nn as nn
import torch.optim as optim
import os
from modelo import RedNeuronalSimple
from datos import cargar_mnist, obtener_info_dataset
from entrenamiento import entrenar_modelo
from utilidades import (guardar_modelo, visualizar_resultados, 
                       mostrar_resumen_modelo, imprimir_metricas_finales)
import configuracion as config


def crear_directorios():
    """Crea los directorios necesarios si no existen."""
    os.makedirs('modelos', exist_ok=True)
    os.makedirs('resultados', exist_ok=True)
    os.makedirs(config.RUTA_DATOS, exist_ok=True)


def configurar_dispositivo():
    """
    Configura el dispositivo (GPU/CPU) para el entrenamiento.
    
    Returns:
        torch.device: Dispositivo configurado
    """
    if config.USAR_GPU and torch.cuda.is_available():
        dispositivo = torch.device('cuda')
        print(f"✓ GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = torch.device('cpu')
        print("✓ Usando CPU")
    
    return dispositivo


def main():
    """Función principal del programa."""
    print("\n" + "="*60)
    print("CLASIFICADOR DE DÍGITOS MNIST")
    print("="*60 + "\n")
    
    # Crear directorios necesarios
    crear_directorios()
    
    # Configurar dispositivo
    dispositivo = configurar_dispositivo()
    
    # Cargar datos
    print("\n Cargando datos...")
    cargador_entrenamiento, cargador_prueba = cargar_mnist(
        ruta_datos=config.RUTA_DATOS,
        tamaño_lote=config.TAMAÑO_LOTE
    )
    obtener_info_dataset(cargador_entrenamiento, cargador_prueba)
    
    # Crear modelo
    print("\n Creando modelo...")
    modelo = RedNeuronalSimple(
        entrada=config.NEURONAS_ENTRADA,
        oculta1=config.NEURONAS_OCULTA1,
        oculta2=config.NEURONAS_OCULTA2,
        salida=config.NEURONAS_SALIDA,
        dropout=config.TASA_DROPOUT
    ).to(dispositivo)
    
    mostrar_resumen_modelo(modelo)
    
    # Configurar función de pérdida y optimizador
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=config.TASA_APRENDIZAJE)
    
    # Entrenar modelo
    print(" Iniciando entrenamiento...")
    historial = entrenar_modelo(
        modelo=modelo,
        cargador_entrenamiento=cargador_entrenamiento,
        cargador_prueba=cargador_prueba,
        criterio=criterio,
        optimizador=optimizador,
        dispositivo=dispositivo,
        epicas=config.EPICAS,
        verbose=True
    )
    
    # Mostrar métricas finales
    imprimir_metricas_finales(historial)
    
    # Guardar modelo
    print(" Guardando modelo...")
    guardar_modelo(modelo, config.RUTA_MODELO)
    
    # Visualizar y guardar resultados
    print("\n Generando gráficas...")
    visualizar_resultados(historial, config.RUTA_GRAFICAS)
    
    print("\n Entrenamiento completado exitosamente!\n")


if __name__ == '__main__':
    main()