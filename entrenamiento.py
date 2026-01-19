import torch


def entrenar_epoca(modelo, cargador, criterio, optimizador, dispositivo, epoca, verbose=True):
    """
    Entrena el modelo durante una época.
    
    Args:
        modelo: Modelo de PyTorch a entrenar
        cargador: DataLoader con los datos de entrenamiento
        criterio: Función de pérdida
        optimizador: Optimizador para actualizar los pesos
        dispositivo: Dispositivo donde ejecutar el entrenamiento (CPU/GPU)
        epoca: Número de época actual
        verbose: Si True, imprime información del progreso (default: True)
        
    Returns:
        tuple: (pérdida_promedio, precisión)
    """
    modelo.train()
    perdida_acumulada = 0.0
    correctos = 0
    total = 0
    
    for indice_lote, (datos, objetivo) in enumerate(cargador):
        # Mover datos al dispositivo
        datos, objetivo = datos.to(dispositivo), objetivo.to(dispositivo)
        
        # Resetear gradientes
        optimizador.zero_grad()
        
        # Propagación hacia adelante
        salida = modelo(datos)
        perdida = criterio(salida, objetivo)
        
        # Propagación hacia atrás y optimización
        perdida.backward()
        optimizador.step()
        
        # Calcular estadísticas
        perdida_acumulada += perdida.item()
        _, prediccion = torch.max(salida.data, 1)
        total += objetivo.size(0)
        correctos += (prediccion == objetivo).sum().item()
        
        # Imprimir progreso
        if verbose and indice_lote % 100 == 0:
            print(f'  Época {epoca}, Lote {indice_lote}/{len(cargador)}, '
                  f'Pérdida: {perdida.item():.4f}')
    
    # Calcular métricas finales
    perdida_promedio = perdida_acumulada / len(cargador)
    precision = 100 * correctos / total
    
    return perdida_promedio, precision


def evaluar_modelo(modelo, cargador, criterio, dispositivo):
    """
    Evalúa el modelo en el conjunto de prueba.
    
    Args:
        modelo: Modelo de PyTorch a evaluar
        cargador: DataLoader con los datos de prueba
        criterio: Función de pérdida
        dispositivo: Dispositivo donde ejecutar la evaluación (CPU/GPU)
        
    Returns:
        tuple: (pérdida_promedio, precisión)
    """
    modelo.eval()
    perdida_total = 0
    correctos = 0
    total = 0
    
    with torch.no_grad():
        for datos, objetivo in cargador:
            # Mover datos al dispositivo
            datos, objetivo = datos.to(dispositivo), objetivo.to(dispositivo)
            
            # Propagación hacia adelante
            salida = modelo(datos)
            perdida_total += criterio(salida, objetivo).item()
            
            # Calcular precisión
            _, prediccion = torch.max(salida.data, 1)
            total += objetivo.size(0)
            correctos += (prediccion == objetivo).sum().item()
    
    # Calcular métricas finales
    perdida_promedio = perdida_total / len(cargador)
    precision = 100 * correctos / total
    
    return perdida_promedio, precision


def entrenar_modelo(modelo, cargador_entrenamiento, cargador_prueba, 
                   criterio, optimizador, dispositivo, epicas=5, verbose=True):
    """
    Entrena el modelo durante múltiples épocas.
    
    Args:
        modelo: Modelo de PyTorch a entrenar
        cargador_entrenamiento: DataLoader con datos de entrenamiento
        cargador_prueba: DataLoader con datos de prueba
        criterio: Función de pérdida
        optimizador: Optimizador para actualizar los pesos
        dispositivo: Dispositivo donde ejecutar (CPU/GPU)
        epicas: Número de épocas de entrenamiento (default: 5)
        verbose: Si True, imprime información del progreso (default: True)
        
    Returns:
        dict: Diccionario con historial de pérdidas y precisiones
    """
    historial = {
        'perdidas_entrenamiento': [],
        'precisiones_entrenamiento': [],
        'perdidas_prueba': [],
        'precisiones_prueba': []
    }
    
    for epoca in range(1, epicas + 1):
        if verbose:
            print(f'\n{"="*60}')
            print(f'Época {epoca}/{epicas}')
            print(f'{"="*60}')
        
        # Entrenar
        perdida_ent, precision_ent = entrenar_epoca(
            modelo, cargador_entrenamiento, criterio, 
            optimizador, dispositivo, epoca, verbose
        )
        
        # Evaluar
        perdida_pru, precision_pru = evaluar_modelo(
            modelo, cargador_prueba, criterio, dispositivo
        )
        
        # Guardar historial
        historial['perdidas_entrenamiento'].append(perdida_ent)
        historial['precisiones_entrenamiento'].append(precision_ent)
        historial['perdidas_prueba'].append(perdida_pru)
        historial['precisiones_prueba'].append(precision_pru)
        
        # Mostrar resultados
        if verbose:
            print(f'\nResultados Época {epoca}:')
            print(f'  Entrenamiento - Pérdida: {perdida_ent:.4f}, Precisión: {precision_ent:.2f}%')
            print(f'  Prueba        - Pérdida: {perdida_pru:.4f}, Precisión: {precision_pru:.2f}%')
    
    return historial