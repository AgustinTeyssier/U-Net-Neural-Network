# U-Net Neural Network for Pneumothorax Segmentation

Este proyecto implementa una red neuronal U-Net para la segmentación de neumotórax en imágenes de rayos X de tórax. Desarrollado como parte de mi exploración en redes convolucionales para aplicaciones médicas.

## Objetivo

Crear un modelo de segmentación semántica capaz de identificar regiones de neumotórax en imágenes JPEG de rayos X, utilizando la arquitectura U-Net clásica.

## Dataset

Utilicé el dataset **"Pneumothorax Chest X-Ray Images and Masks"** de Kaggle, que contiene imágenes en formato JPEG con sus respectivas máscaras binarias en PNG.

## Implementación

La arquitectura fue construida usando TensorFlow/Keras, con las siguientes características:
- Capas de convolución y max-pooling en el encoder
- Skip connections para preservar información espacial
- Métricas de evaluación: Coeficiente Dice e IoU (Intersection over Union)
- Preprocesamiento específico para imágenes JPEG médicas

## Resultados

El modelo logra segmentar eficientemente las regiones de neumotórax, con métricas que demuestran su capacidad para aplicaciones de asistencia al diagnóstico médico.

## Uso

Para ejecutar el proyecto:
```bash
code SegundaUNet.py
##Modificar el numero de epocas ( epochs = 50) 
python3 SegundaUNet.py
