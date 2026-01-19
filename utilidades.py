import torch
import matplotlib.pyplot as plt


def guardar_modelo(modelo, ruta='modelos/modelo_mnist.pth'):
    """
    Guarda el estado del modelo en un archivo.
    
    Args:
        modelo: Modelo de PyTorch a guardar
        ruta: Ruta donde guardar el modelo (default: 'modelos/modelo_mnist.pth')
    """
    torch.save(modelo.state_dict(), ruta)
    print(f"Modelo guardado en '{ruta}'")


def cargar_modelo(modelo, ruta='modelos/modelo_mnist.pth', dispositivo='cpu'):
    """
    Carga el estado del modelo desde un archivo.
    
    Args:
        modelo: Modelo de PyTorch donde cargar los pesos
        ruta: Ruta del archivo del modelo (default: 'modelos/modelo_mnist.pth')
        dispositivo: Dispositivo donde cargar el modelo (default: 'cpu')
        
    Returns:
        Modelo con los pesos cargados
    """
    modelo.load_state_dict(torch.load(ruta, map_location=dispositivo))
    print(f"Modelo cargado desde '{ruta}'")
    return modelo


def visualizar_resultados(historial, guardar_ruta='resultados/graficas_entrenamiento.png'):
    """
    Crea y guarda gráficas de pérdida y precisión.
    
    Args:
        historial: Diccionario con el historial de entrenamiento
        guardar_ruta: Ruta donde guardar las gráficas (default: 'resultados/graficas_entrenamiento.png')
    """
    plt.figure(figsize=(14, 5))
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(historial['perdidas_entrenamiento'], 
             label='Pérdida Entrenamiento', marker='o', linewidth=2)
    plt.plot(historial['perdidas_prueba'], 
             label='Pérdida Prueba', marker='s', linewidth=2)
    plt.xlabel('Épica', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.title('Evolución de la Pérdida', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(historial['precisiones_entrenamiento'], 
             label='Precisión Entrenamiento', marker='o', linewidth=2)
    plt.plot(historial['precisiones_prueba'], 
             label='Precisión Prueba', marker='s', linewidth=2)
    plt.xlabel('Épica', fontsize=12)
    plt.ylabel('Precisión (%)', fontsize=12)
    plt.title('Evolución de la Precisión', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(guardar_ruta, dpi=300, bbox_inches='tight')
    print(f"Gráficas guardadas en '{guardar_ruta}'")
    plt.show()


def mostrar_resumen_modelo(modelo):
    """
    Muestra un resumen de la arquitectura del modelo.
    
    Args:
        modelo: Modelo de PyTorch
    """
    print("\n" + "="*60)
    print("RESUMEN DEL MODELO")
    print("="*60)
    print(modelo)
    print("="*60)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    
    print(f"\nParámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print("="*60 + "\n")


def imprimir_metricas_finales(historial):
    """
    Imprime las métricas finales del entrenamiento.
    
    Args:
        historial: Diccionario con el historial de entrenamiento
    """
    print("\n" + "="*60)
    print("MÉTRICAS FINALES")
    print("="*60)
    print(f"Mejor precisión en entrenamiento: {max(historial['precisiones_entrenamiento']):.2f}%")
    print(f"Mejor precisión en prueba: {max(historial['precisiones_prueba']):.2f}%")
    print(f"Pérdida final en entrenamiento: {historial['perdidas_entrenamiento'][-1]:.4f}")
    print(f"Pérdida final en prueba: {historial['perdidas_prueba'][-1]:.4f}")
    print("="*60 + "\n")