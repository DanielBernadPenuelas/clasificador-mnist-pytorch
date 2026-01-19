import torch.nn as nn


class RedNeuronalSimple(nn.Module):
    """
    Red neuronal simple para clasificación de dígitos MNIST.
    
    Arquitectura:
    - Capa de entrada: 784 neuronas (28x28 píxeles aplanados)
    - Capa oculta 1: 128 neuronas + ReLU + Dropout(0.2)
    - Capa oculta 2: 64 neuronas + ReLU + Dropout(0.2)
    - Capa de salida: 10 neuronas (clases 0-9)
    """
    
    def __init__(self, entrada=784, oculta1=128, oculta2=64, salida=10, dropout=0.2):
        """
        Inicializa la red neuronal.
        
        Args:
            entrada: Número de características de entrada (default: 784)
            oculta1: Neuronas en la primera capa oculta (default: 128)
            oculta2: Neuronas en la segunda capa oculta (default: 64)
            salida: Número de clases de salida (default: 10)
            dropout: Tasa de dropout (default: 0.2)
        """
        super(RedNeuronalSimple, self).__init__()
        self.aplanar = nn.Flatten()
        self.red = nn.Sequential(
            nn.Linear(entrada, oculta1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(oculta1, oculta2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(oculta2, salida)
        )
    
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida con las predicciones
        """
        x = self.aplanar(x)
        return self.red(x)