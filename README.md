# Détection d'objets en temps réel – YOLOv8 et OpenCV

Application de **vision par ordinateur** qui détecte et identifie des objets en temps réel à partir de la webcam ou d'une vidéo. Le modèle YOLOv8 repère les objets, et OpenCV affiche les résultats : boîte colorée autour de chaque objet, nom de la classe, niveau de confiance, nombre d'objets et FPS.

## Fonctionnalités

- Détection d'objets en temps réel (webcam ou fichier vidéo)
- Reconnaissance de 80 classes d'objets du jeu de données COCO (personne, voiture, chien, feu de circulation, téléphone, etc.)
- Affichage du nom, du niveau de confiance, du nombre d'objets et des FPS
- Capture d'écran en appuyant sur une touche
- Paramètres réglables en ligne de commande

## Technologies

- Python 3
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) (modèle `yolov8n.pt`)
- OpenCV (`opencv-python`)

## Installation

```bash
git clone https://github.com/fallmoda22-web/Detection-Objets-YOLO.git
cd Detection-Objets-YOLO
pip install -r requirements.txt
```

Au premier lancement, le modèle `yolov8n.pt` est téléchargé automatiquement (connexion Internet nécessaire).

## Utilisation

```bash
python object_detection.py
```

| Touche | Action |
|--------|--------|
| `q` | Quitter |
| `s` | Enregistrer une capture d'écran |

Exemples avec options :

```bash
# Autre caméra et seuil de confiance plus élevé
python object_detection.py --source 1 --conf 0.6

# Détection sur une vidéo
python object_detection.py --source ma_video.mp4
```

| Option | Rôle | Défaut |
|--------|------|--------|
| `--source` | Indice de la caméra ou chemin d'une vidéo | `0` |
| `--model` | Modèle YOLO à utiliser | `yolov8n.pt` |
| `--conf` | Seuil de confiance (0 à 1) | `0.5` |
| `--imgsz` | Taille d'entrée du modèle (pixels) | `640` |

## Fonctionnement

1. Lecture de chaque image de la caméra ou de la vidéo avec OpenCV.
2. Détection des objets par le modèle YOLOv8.
3. Dessin des boîtes, des noms de classes et des niveaux de confiance sur l'image.
4. Affichage du nombre d'objets et des FPS.
5. Affichage en temps réel dans une fenêtre.

## Limites et améliorations possibles

- Le modèle `yolov8n.pt` est rapide mais moins précis que les modèles plus grands (`yolov8s.pt`, `yolov8m.pt`).
- Les performances dépendent de l'ordinateur ; un GPU accélère fortement la détection.
- Amélioration possible : entraîner le modèle sur ses propres images pour reconnaître des objets spécifiques.
- Extension possible : suivi des objets, comptage, alertes.

## Auteur

**Moda Fall** – Étudiant en Licence Robotique, Université Numérique Cheikh Hamidou Kane
[GitHub](https://github.com/fallmoda22-web) · [LinkedIn](https://www.linkedin.com/in/moda-fall-95a874344)
