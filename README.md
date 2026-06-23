# Classification d'images CIFAR-10 — ML classiques, CNN et Transfer Learning

Projet réalisé dans le cadre du cours **IA02 – Intelligence Artificielle** (Université de Technologie de Troyes, promotion SN2).

Comparaison de **2 algorithmes de Machine Learning classiques**, **3 architectures CNN** et **1 CNN hybride par Transfer Learning** sur le dataset CIFAR-10.

> Ce dépôt correspond à la **Partie 1** du projet. La Partie 2 (détection de fractures osseuses sur le dataset FracAtlas) se trouve dans un [dépôt séparé](https://github.com/<votre-pseudo>/fracatlas-fracture-detection).

## Auteurs

- Yves CHEKOUA
- Mohamed Mehdi TRABELSSI

## Le dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) contient 60 000 images couleur de 32×32 pixels, réparties en 10 classes équilibrées (avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion) : 50 000 images d'entraînement et 10 000 de test.

![Exemples par classe](images/les_classes.png)

## Algorithmes comparés

| Catégorie | Modèle | Exactitude (test) |
|---|---|---|
| ML classique | SVM (noyau RBF) | 47,63 % |
| ML classique | Régression Logistique | ~40 % |
| ML classique | Random Forest | 47,73 % |
| CNN | CNN Simple (baseline) | 70,94 % |
| CNN | CNN Intermédiaire | 72,50 % |
| CNN | CNN Avancé (régularisé) | 81,66 % |
| CNN Hybride | MobileNetV2 (Transfer Learning) | **87,25 %** |

Détails complets, méthodologie et analyse : voir [RAPPORT.md](RAPPORT.md).

## Structure du dépôt

```
cifar10-classification/
├── cifar10_classification.ipynb   # Notebook complet (tous les modèles)
├── images/                        # Visualisations (matrices de confusion, courbes...)
├── README.md
└── RAPPORT.md                     # Rapport détaillé (méthodologie, résultats, analyse)
```

## Exécution

Le notebook est conçu pour être exécuté sur **Google Colab** (GPU recommandé pour les CNN).

1. Ouvrir `cifar10_classification.ipynb` dans Google Colab
2. Exécuter les cellules dans l'ordre (le dataset CIFAR-10 est chargé directement via `tensorflow.keras.datasets`)
3. Les tailles de données ont été réduites (10 000 images pour la recherche d'hyperparamètres, sous-échantillons pour les CNN hybrides) afin de limiter les temps de calcul sur CPU — voir le rapport pour la justification

### Dépendances

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn opencv-python
```

## Principaux enseignements

- Le **Transfer Learning** (MobileNetV2) surpasse systématiquement les CNN entraînés de zéro, même avec peu de données
- La **régularisation** (BatchNormalization + Dropout) apporte un gain de +9 points d'exactitude par rapport à une architecture CNN équivalente sans régularisation
- Les algorithmes ML classiques (SVM, RF) plafonnent autour de 48 % sur des pixels bruts aplatis : ils ne capturent pas la structure spatiale des images
- Redimensionner les images à 96×96 (au lieu de 32×32) est indispensable pour exploiter MobileNetV2 sans effondrement des feature maps

## Licence

Projet académique — UTT, IA02, 2026.
