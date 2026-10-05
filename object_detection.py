"""Détection d'objets en temps réel avec YOLOv8 et OpenCV.

Touches :
    q  -> quitter
    s  -> enregistrer une capture d'écran
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Détection d'objets en temps réel (YOLOv8 + OpenCV)"
    )
    parser.add_argument("--source", default="0",
                        help="indice de la caméra (0 par défaut) ou chemin d'une vidéo")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="modèle YOLO (yolov8n.pt = léger et rapide)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="seuil de confiance entre 0 et 1 (0.5 par défaut)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="taille d'entrée du modèle en pixels")
    return parser.parse_args()


def color_for(class_id):
    """Couleur stable pour chaque classe."""
    return ((37 * class_id) % 255, (91 * class_id + 80) % 255, (151 * class_id + 160) % 255)


def main():
    args = parse_args()

    # Le modèle est téléchargé automatiquement au premier lancement
    model = YOLO(args.model)
    names = model.names

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la source : {args.source}")

    prev_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fin de la vidéo ou plus d'image reçue.")
            break

        # Détection
        result = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        # Dessin des boîtes et des étiquettes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            color = color_for(class_id)
            label = f"{names[class_id]} {confidence:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # FPS lissés
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        cv2.putText(frame, f"Objets : {len(result.boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS : {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Detection d'objets - YOLOv8 + OpenCV", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(name, frame)
            print(f"Capture enregistrée : {name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
