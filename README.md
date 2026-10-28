# Classification de nodules pulmonaires sur scanners CT (LUNA16)

Mini-projet 14 – Master Ingénierie Digitale pour les métiers de la Santé, ENSAM Casablanca (2025–2026).

> **Preuve de concept à visée pédagogique, non validée cliniquement.** Lire la section [Limites](#limites) avant d'interpréter les résultats.

## Objectif

Comparer trois approches de deep learning pour classer des patchs de scanners thoraciques en **nodule / non-nodule** :

1. CNN 3D utilisé comme extracteur de features + Random Forest
2. U-Net avec backbone VGG16 (transfer learning)
3. YOLOv8

## Données

- **LUNA16** (dérivé de LIDC-IDRI), scanners CT au format MetaImage (`.mhd` / `.raw`), lus avec SimpleITK.
  Source utilisée : <https://www.kaggle.com/datasets/tessnimetteib/luna16-dataset>
- Les données **ne sont pas incluses** dans ce dépôt (taille et licence).
- 242 scanners traités sur 601 disponibles (359 écartés pour valeurs HU invalides).

## Pipeline de prétraitement

1. Lecture du scanner (SimpleITK) et application de slope / intercept → unités Hounsfield (HU)
2. Clipping des HU dans [-1000, +400]
3. Normalisation Z-score
4. Conversion des coordonnées monde (mm) → voxels à partir de l'origine et du spacing
5. Extraction de patchs 3D de 32×48×48 voxels
   - 465 positifs centrés sur les nodules annotés
   - 968 négatifs tirés au hasard à au moins 15 voxels de tout nodule (4 par scanner)
6. Équilibrage par flip horizontal de 503 positifs → 968 positifs / 968 négatifs (1 936 patchs)
7. Découpage 70 / 15 / 15 : 1 354 train, 291 validation, 291 test

## Modèles

| Modèle | Entrée | Principe |
|---|---|---|
| CNN 3D + Random Forest | patch 3D (32×48×48) | 3 blocs Conv3D, features GAP 128-D, classifieur Random Forest (303 297 paramètres) |
| U-Net + VGG16 | coupe 2D centrale (48×48) | encodeur VGG16 gelé (ImageNet), masques circulaires synthétiques (rayon 8 px) |
| YOLOv8 | coupe 2D centrale redimensionnée en 64×64 | boîte centrée de taille fixe, AdamW, 25 epochs |

## Résultats

Jeu de test : 291 patchs (145 nodules, 146 non-nodules).

| Modèle | Exactitude | Rappel | Spécificité | AUC |
|---|---|---|---|---|
| CNN 3D + Random Forest | 91,07 % | 95,86 % | 86,30 % | 0,9613 |
| U-Net + VGG16 | 92,10 % | 99,31 % | 84,93 % | 0,9885 |
| YOLOv8 | 96,91 % | 99,31 % | 94,52 % | 0,9929 |

Ces chiffres sont à lire avec les limites ci-dessous : ils ne mesurent pas une performance de détection réelle sur des scanners complets.

## Limites

1. **Découpage au niveau des patchs, après augmentation.** Un nodule et sa copie retournée, ou deux patchs d'un même scanner, peuvent se retrouver en train et en test : risque de fuite de données, donc de résultats optimistes.
2. **Négatifs faciles et test équilibré.** Les négatifs sont des positions aléatoires, pas les candidats difficiles de `candidates.csv`. Le ratio réel de ce fichier est d'environ 407 négatifs pour 1 positif, très loin du 50/50 du test.
3. **Classification de patchs, pas détection de bout en bout.** U-Net et YOLOv8 travaillent sur une seule coupe 2D ; les masques du U-Net sont synthétiques ; la boîte YOLOv8 est toujours centrée. Pas de métrique FROC (métrique officielle du challenge).
4. **Sous-ensemble des données** : 242 scanners sur 601 exploités.
5. **YOLOv8** : surapprentissage observé après l'epoch 17 (mAP50 qui chute).
6. **Pas de validation externe.**

## Pistes d'amélioration

- Découpage par scanner ou par sous-ensemble LUNA16 (subset), augmentation uniquement sur le train
- Négatifs issus de `candidates.csv` et évaluation en FROC
- Modèles 3D de bout en bout (3D U-Net) et explicabilité (Grad-CAM)
- Validation sur un jeu externe

## Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/        notebook du pipeline complet (Kaggle, GPU Tesla T4)
└── report/
    └── LUNA16_Report.pdf   rapport complet (25 pages, en anglais)
```

## Reproduire

Les expériences ont été menées sur Kaggle Notebooks (GPU Tesla T4, Python 3.12, TensorFlow / Keras, Ultralytics YOLOv8).

```bash
pip install -r requirements.txt
```

Télécharger LUNA16 (lien ci-dessus), puis exécuter `notebooks/luna16_pipeline.ipynb` (sections décrites dans `notebooks/README.md`).

## Auteure

Siham AIT TALEB – Master Ingénierie Digitale pour les métiers de la Santé, ENSAM Casablanca.
