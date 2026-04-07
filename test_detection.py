import cv2
import os
from ultralytics import YOLO

MODEL_EGG   = r"models\egg_detection\weights\best.pt"
MODEL_BARST = r"models\crack_detection\weights\best.pt"

TEST_EGG   = r"datasets\egg_detection\test\images"
TEST_BARST = r"datasets\crack_detection\test\images"

model_egg   = YOLO(MODEL_EGG)
model_barst = YOLO(MODEL_BARST)

print("Kies testset:")
print("  1 = egg detectie")
print("  2 = barst detectie")
keuze = input("Keuze (1/2): ").strip()

if keuze == '1':
    folder = TEST_EGG
    modus = 'egg'
else:
    folder = TEST_BARST
    modus = 'barst'

images = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
print(f"\n{len(images)} foto's gevonden. Navigeer met PIJLTJES, ESC om te stoppen.\n")

idx = 0
cv2.namedWindow("Test detectie", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Test detectie", 1280, 720)

while True:
    img_path = os.path.join(folder, images[idx])
    img = cv2.imread(img_path)
    display = img.copy()

    if modus == 'egg':
        res = model_egg(img, conf=0.65, verbose=False)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"egg {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        n = len(res.boxes)

    else:
        res = model_barst(img, conf=0.65, verbose=False)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = res.names[int(box.cls[0])]
            kleur = (0, 0, 255) if label == "Crack" else (255, 165, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), kleur, 2)
            cv2.putText(display, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, kleur, 1)
        n = len(res.boxes)

    # Info bovenaan
    info = f"[{idx+1}/{len(images)}] {images[idx]}  |  {n} detecties"
    cv2.putText(display, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Test detectie", display)
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key in (83, 84, ord('d'), ord(' ')):  # pijl rechts / D / spatie
        idx = min(idx + 1, len(images) - 1)
    elif key in (81, 82, ord('a')):  # pijl links / A
        idx = max(idx - 1, 0)

cv2.destroyAllWindows()
