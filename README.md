# Clasificacion-de-comando-de-voz-ONLINE

## Descripción
Este proyecto tiene como objetivo clasificar comandos de voz en tiempo real, utilizando un modelo de aprendizaje profundo (Deep Learning) y una red neuronal convolucional (CNN). El modelo fue entrenado con 8 diferentes de comandos de voz.

| Word        | Precision              | Recall |
|-------------|------------------------|--------|
| temperatura | 0.8                    | 0.8    |
| tarea       | 1                      | 0.7    |
| avisos      | 0.8888888888888888     | 0.8    |
| alemania    | 0.9                    | 0.9    |
| fotografiar | 0.6666666666666666     | 0.8    |
| onda        | 0.9                    | 0.9    |
| mes         | 0.8181818181818182     | 0.9    |
| rascar      | 0.9090909090909091     | 1      |

## Instalación
### Clonar el repositorio:
```bash
    git clone https://github.com/Pau025/Clasificacion-de-comando-de-voz-ONLINE.git
```

### Para instalar las librerías necesarias, ejecutar el siguiente comando:
```bash
    pip install -r requirements.txt
```

> [!WARNING]  
> Asegúrate de instalar esto en un entorno virtual o sobrescribirá todos los paquetes en tu instalación global de Python, incluyendo la eliminación de todas las dependencias no listadas en este archivo. Configura un entorno virtual de Python de esta manera

## Uso
Para ejecutar el programa, ejecutar el siguiente comando:
```bash
    streamlit run quickstart.py
```
