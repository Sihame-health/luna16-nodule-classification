# Notebook

`luna16_pipeline.ipynb` : pipeline complet exécuté sur Kaggle (GPU Tesla T4), dans l'ordre :

1. Exploration du dataset LUNA16 (annotations, candidats, répartition par sous-ensemble)
2. Prétraitement (unités Hounsfield, clipping, normalisation) et extraction des patchs 3D
3. Équilibrage par augmentation (flip) des patchs positifs
4. CNN 3D + Random Forest
5. U-Net + VGG16
6. YOLOv8
