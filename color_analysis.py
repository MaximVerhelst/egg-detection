import cv2
import numpy as np
import os
from ultralytics import YOLO

MODEL_PATH = r"models\egg_detection\weights\best.pt"
IMAGE_DIR  = r"data_collection\images\eggs"

model = YOLO(MODEL_PATH)
images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))])
print(f"{len(images)} foto's gevonden. Navigeer met D/spatie (volgende) en A (vorige). ESC = stoppen.")


def dominant_kleur_kmeans(pixels_bgr, k=3):
    pixels = np.float32(pixels_bgr)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    return centers[np.argmax(counts)].astype(np.uint8)


def classificeer_kleur(bgr):
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    pixel = np.uint8([[bgr]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 60:
        return "?", h, s, v
    elif r > 185 and b > 185 and g > 185 and abs(r - b) < 30 and abs(r - g) < 30:
        return "WIT", h, s, v
    else:
        return "BRUIN", h, s, v


def analyseer(img_path):
    img = cv2.imread(img_path)
    results = model(img_path, conf=0.5, verbose=False)[0]
    result_img = np.zeros_like(img)
    egg_data = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx, ry = (x2 - x1) // 2, (y2 - y1) // 2

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        pixels = img[mask == 255]
        dominant_bgr = dominant_kleur_kmeans(pixels, k=3)
        kleur, h, s, v = classificeer_kleur(dominant_bgr)

        egg_data.append({'i': i, 'kleur': kleur, 'h': h, 's': s, 'v': v,
                         'bgr': dominant_bgr, 'cx': cx, 'cy': cy})

        result_img[mask == 255] = img[mask == 255]
        cv2.putText(result_img, f"#{i+1}", (cx - 20, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2, cv2.LINE_AA)

        rand_kleur = (255, 255, 255) if kleur == "WIT" else (0, 140, 255) if kleur == "BRUIN" else (0, 0, 255)
        cv2.ellipse(result_img, (cx, cy), (rx, ry), 0, 0, 360, rand_kleur, 2)

    # Info paneel
    info_h = result_img.shape[0]
    info_w = 320
    info_panel = np.zeros((info_h, info_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(info_panel, "#   Kleur    H    S    V", (8, 28), font, 0.60, (255, 255, 255), 2)
    cv2.line(info_panel, (5, 36), (info_w - 5, 36), (80, 80, 80), 1)

    wit = sum(1 for r in egg_data if r['kleur'] == "WIT")
    bruin = sum(1 for r in egg_data if r['kleur'] == "BRUIN")
    cv2.putText(info_panel, f"WIT:{wit}  BRUIN:{bruin}", (8, info_h - 10),
                font, 0.55, (200, 200, 200), 1)

    for r in egg_data:
        y = 56 + r['i'] * 26
        if y > info_h - 30:
            break
        tint = (255, 255, 255) if r['kleur'] == "WIT" else (0, 140, 255) if r['kleur'] == "BRUIN" else (80, 80, 80)
        tekst = f"{r['i']+1:<3} {r['kleur']:<7} {r['h']:<4} {r['s']:<4} {r['v']}"
        cv2.putText(info_panel, tekst, (8, y), font, 0.55, tint, 1, cv2.LINE_AA)
        b, g, re = int(r['bgr'][0]), int(r['bgr'][1]), int(r['bgr'][2])
        cv2.rectangle(info_panel, (info_w - 28, y - 16), (info_w - 5, y + 4), (b, g, re), -1)

    return np.hstack([result_img, info_panel]), len(egg_data), egg_data


cv2.namedWindow("Kleurdetectie", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Kleurdetectie", 1400, 720)

idx = 0
while True:
    img_path = os.path.join(IMAGE_DIR, images[idx])
    combined, n, egg_data = analyseer(img_path)

    info = f"[{idx+1}/{len(images)}] {images[idx]}  |  {n} eieren"
    cv2.putText(combined, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Print waarden van '?' eieren
    for r in egg_data:
        if r['kleur'] == "?":
            b, g, re = int(r['bgr'][0]), int(r['bgr'][1]), int(r['bgr'][2])
            print(f"  ? #{r['i']+1}: B:{b} G:{g} R:{re}  H:{r['h']} S:{r['s']} V:{r['v']}")

    cv2.imshow("Kleurdetectie", combined)
    key = cv2.waitKey(0)

    if key == 27:
        break
    elif key in (83, 84, ord('d'), ord(' ')):
        idx = min(idx + 1, len(images) - 1)
    elif key in (81, 82, ord('a')):
        idx = max(idx - 1, 0)

cv2.destroyAllWindows()
