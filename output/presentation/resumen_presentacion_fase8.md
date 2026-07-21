# Resumen de presentacion - Colorectal AI

## Dataset
- 100,000 imagenes TIFF 224x224.
- Split: train 79,994; validation 9,995; test 10,011.
- Sin corruptas ni duplicados exactos.

## Evaluacion demo
| Modelo | Accuracy | Macro F1 | Macro AUC | MCC |
| --- | ---: | ---: | ---: | ---: |
| cnn_simple | 44.44% | 0.5644 | 0.8300 | 0.3872 |
| best_model | 11.11% | 0.2000 | 0.5000 | 0.0000 |
| mobilenetv2 | 11.11% | 0.2000 | 0.5000 | 0.0000 |
| hybrid_attention | 11.11% | 0.2000 | 0.4022 | 0.0000 |
| hybrid_autoencoder | 11.11% | 0.2000 | 0.4789 | 0.0000 |
| resnet50v2 | 2.22% | 0.0909 | 0.4761 | -0.1107 |

Interpretacion: CNN Simple lidera en la muestra demo. Estos valores sirven para mostrar el pipeline, no como conclusion clinica final.

## Pruebas estadisticas
- McNemar p-value: `0.0013189686125793583`.
- DeLong p-value para TUM: `4.475502407660346e-05`.
- Diferencia pareada de accuracy: `-0.333333`.

Interpretacion: en la muestra demo, la diferencia entre CNN Simple y best_model es significativa; debe escalarse para conclusion final.

## Validacion cruzada demo
| Modelo | Accuracy mean | Accuracy std | Macro F1 mean | Macro AUC mean | MCC mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| cnn_simple | 48.89% | 0.0648 | 0.7203 | 0.8326 | 0.4569 |
| best_model | 11.11% | 0.0000 | 0.2000 | 0.5000 | 0.0000 |

Interpretacion: los folds K=5 existen y estan balanceados. La demo usa modelos ya entrenados; la CV cientifica final debe reentrenar por fold.