# Fase 5 - Pruebas estadisticas pareadas

- Comparacion: `best_model` vs `cnn_simple`
- Clase usada para DeLong/AUC: `TUM`
- Artefactos: mismas muestras, etiquetas, particion y fingerprint validados por el backend.
- Registro backend: `E:\CICLO07\ING. SOFTWARE\PROY3\cancer-colorrectal\experiments\reports\paired_model_comparison_20260721T092857_319749Z.json`

## Resultado compacto

- McNemar p-value: `0.0013189686125793583`
- McNemar significativo: `True`
- Diferencia de accuracy pareada: `-0.333333`
- IC 95% accuracy diff: `-0.478334` a `-0.188333`
- Diferencia AUC DeLong: `-0.31`
- DeLong p-value: `4.475502407660346e-05`

Nota: esta comparacion usa la muestra demo urgente de Fase 4; sirve para demostrar el flujo estadistico, no para concluir desempeno clinico final.