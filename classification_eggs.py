import pyzed.sl as sl
import cv2
import numpy as np
import json
import os
import subprocess
import sys
from ultralytics import YOLO

MODEL_EGG   = r"models\egg_detection\weights\best.pt"
MODEL_BARST = r"models\crack_detection\weights\best.pt"
KALI_PATH   = r"calibration.json"
KALI_SCRIPT = r"calibrate.py"

EXPOSURE   = 15
GAIN       = 35
SHARPNESS  = 8
CONTRAST   = 7
SATURATION = 5
WHITBALANS = 4400

# --- Kalibratie ---
keuze = input("Wil je kalibreren? (j/n): ").strip().lower()
if keuze == 'j':
    print("Kalibratie wordt gestart...")
    subprocess.run([sys.executable, KALI_SCRIPT])

if not os.path.exists(KALI_PATH):
    print(f"Geen kalibratie gevonden ({KALI_PATH}). Voer eerst kalibratie uit.")
    exit()

with open(KALI_PATH) as f:
    kali = json.load(f)

fx            = kali['fx']
fy            = kali['fy']
plane_coeffs  = tuple(kali['plane'])
karton_offset = kali['karton_offset']
print(f"Kalibratie geladen. fx={fx:.1f}  fy={fy:.1f}")


def ref_depth(u, v):
    a, b, c = plane_coeffs
    return a * u + b * v + c + karton_offset


# --- Modellen laden ---
print("Modellen laden...")
model_egg   = YOLO(MODEL_EGG)
model_barst = YOLO(MODEL_BARST)

# --- Camera ---
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD2K
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.MILLIMETER
init_params.depth_minimum_distance = 100

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Camera kon niet openen")
    exit()

zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,                 EXPOSURE)
zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN,                     GAIN)
zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS,                SHARPNESS)
zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST,                 CONTRAST)
zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION,               SATURATION)
zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO,        0)
zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, WHITBALANS)

runtime_params = sl.RuntimeParameters()
image_zed = sl.Mat()
depth_map = sl.Mat()


# --- Hulpfuncties ---
def get_median_depth(dmap, cx, cy, r=5):
    vals = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            _, v = dmap.get_value(cx + dx, cy + dy)
            if np.isfinite(v) and v > 0:
                vals.append(v)
    return float(np.median(vals)) if vals else None


def dominant_kleur_kmeans(pixels_bgr, k=3):
    pixels = np.float32(pixels_bgr)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    return centers[np.argmax(counts)].astype(np.uint8)


def classificeer_kleur(bgr):
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    if r > 185 and b > 185 and g > 185 and abs(r - b) < 30 and abs(r - g) < 30:
        return "WIT"
    return "BRUIN"


def classificeer_maat(vol_cm3):
    if vol_cm3 is None:
        return "--"
    if vol_cm3 < 40:
        return "S"
    elif vol_cm3 < 50:
        return "M"
    else:
        return "L"


def barst_per_ei(frame):
    """Detecteer barsten en intacte eieren op volledig frame."""
    res = model_barst(frame, conf=0.65, verbose=False)[0]
    barst_boxes  = []
    intact_boxes = []
    for box in res.boxes:
        label = res.names[int(box.cls[0])]
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if label == "Crack":
            barst_boxes.append((bx1, by1, bx2, by2, conf))
        elif label == "Intact":
            intact_boxes.append((bx1, by1, bx2, by2, conf))
    return barst_boxes, intact_boxes


def best_overlap_conf(ex1, ey1, ex2, ey2, boxes):
    """Geef hoogste conf van overlappende box, of 0 als geen overlap."""
    best = 0.0
    for bx1, by1, bx2, by2, conf in boxes:
        ix1, iy1 = max(ex1, bx1), max(ey1, by1)
        ix2, iy2 = min(ex2, bx2), min(ey2, by2)
        if ix2 > ix1 and iy2 > iy1 and conf > best:
            best = conf
    return best


# --- Venster ---
cv2.namedWindow("Ei Analyse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ei Analyse", 1600, 720)

print("\nGestart. ESC = stoppen.")

while True:
    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        continue

    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

    frame = image_zed.get_data()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    display = frame.copy()

    # --- Egg detectie + crack detectie op volledig frame ---
    results = model_egg(frame, conf=0.3, verbose=False)[0]
    boxes_sorted = sorted(results.boxes, key=lambda b: (int(b.xyxy[0][1]) // 100, int(b.xyxy[0][0])))
    barst_boxes, intact_boxes = barst_per_ei(frame)

    egg_data = []

    for i, box in enumerate(boxes_sorted):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
        egg_conf = float(box.conf[0])

        # --- fitEllipse op contour van ei ---
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fit_ok = False
        if contours and len(contours[0]) >= 5:
            ellipse = cv2.fitEllipse(contours[0])
            (ecx, ecy), (ea, eb), angle = ellipse
            ecx, ecy = int(ecx), int(ecy)
            # ea = grote as diameter, eb = kleine as diameter
            half_a = int(max(ea, eb) / 2)  # halve grote as
            half_b = int(min(ea, eb) / 2)  # halve kleine as (b)
            fit_ok = True

        # --- Kleur ---
        pixels = frame[mask == 255]
        if len(pixels) > 10:
            dom_bgr = dominant_kleur_kmeans(pixels, k=3)
            kleur = classificeer_kleur(dom_bgr)
        else:
            kleur = "?"

        # --- Diepte & volume ---
        d_ei = get_median_depth(depth_map, cx, cy)
        if d_ei is not None and fit_ok:
            d_karton   = ref_depth(cx, cy)
            hoogte_ei  = d_karton - d_ei
            breedte_mm = min(ea, eb) * d_ei / fx  # korte diameter via fitEllipse
            a = hoogte_ei / 2
            b = breedte_mm / 2
            vol_cm3 = (4/3) * np.pi * a * b**2 / 1000 if a > 0 and b > 0 else None
        else:
            hoogte_ei  = None
            breedte_mm = None
            vol_cm3    = None

        # --- Crack detectie ---
        barst_conf  = best_overlap_conf(x1, y1, x2, y2, barst_boxes)
        intact_conf = best_overlap_conf(x1, y1, x2, y2, intact_boxes)
        barst = barst_conf > 0

        maat = classificeer_maat(vol_cm3)

        egg_data.append({
            'i': i, 'kleur': kleur, 'maat': maat,
            'breedte': breedte_mm, 'hoogte': hoogte_ei, 'vol': vol_cm3,
            'barst': barst, 'barst_conf': barst_conf, 'intact_conf': intact_conf, 'egg_conf': egg_conf,
        })

        # --- Tekenen ---
        if barst:
            rand_kleur = (0, 0, 255)
        elif kleur == "WIT":
            rand_kleur = (255, 255, 255)
        else:
            rand_kleur = (0, 140, 255)

        if fit_ok:
            cv2.ellipse(display, ellipse, rand_kleur, 2)

            # OpenCV angle = rotatie van de grote as t.o.v. verticale as
            # Korte as staat loodrecht op grote as
            # Grote as richting: angle graden t.o.v. verticaal = (angle - 90) t.o.v. horizontaal
            groot_rad = np.deg2rad(angle - 90)
            klein_rad = groot_rad + np.pi / 2  # loodrecht = korte as

            half_b_draw = int(min(ea, eb) / 2)
            bx = int(np.cos(klein_rad) * half_b_draw)
            by = int(np.sin(klein_rad) * half_b_draw)
            cv2.line(display, (ecx - bx, ecy - by), (ecx + bx, ecy + by), (0, 255, 255), 3)
            cv2.putText(display, "b", (ecx + bx + 5, ecy + by + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.ellipse(display, (cx, cy), (rx, ry), 0, 0, 360, rand_kleur, 2)

        cv2.putText(display, f"#{i+1}", (cx - 20, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2, cv2.LINE_AA)
        cv2.circle(display, (cx, cy), 4, (0, 255, 255), -1)

    # --- Info paneel ---
    info_h = display.shape[0]
    info_w = 780
    font = cv2.FONT_HERSHEY_SIMPLEX
    info_panel = np.zeros((info_h, info_w, 3), dtype=np.uint8)

    cv2.putText(info_panel, "#  Kleur  Maat  Breed  Hoog  Vol   Barst", (8, 32),
                font, 0.96, (255, 255, 255), 2)
    cv2.line(info_panel, (5, 44), (info_w - 5, 44), (80, 80, 80), 1)

    wit     = sum(1 for e in egg_data if e['kleur'] == "WIT")
    bruin   = sum(1 for e in egg_data if e['kleur'] == "BRUIN")
    barsten = sum(1 for e in egg_data if e['barst'])
    cv2.putText(info_panel, f"WIT:{wit}  BRUIN:{bruin}  BARST:{barsten}",
                (8, info_h - 10), font, 0.96, (200, 200, 200), 1)

    for e in egg_data:
        y = 96 + e['i'] * 48
        if y > info_h - 40:
            break

        # Achtergrondkleur: rood = barst, groen = geen barst
        bg_kleur = (0, 0, 180) if e['barst'] else (0, 120, 0)
        cv2.rectangle(info_panel, (4, y - 28), (info_w - 4, y + 14), bg_kleur, -1)

        kleur_str = e['kleur']
        maat_str  = e['maat']
        breed_str = f"{e['breedte']:.0f}" if e['breedte'] else "--"
        hoog_str  = f"{e['hoogte']:.0f}"  if e['hoogte']  else "--"
        vol_str   = f"{e['vol']:.1f}"     if e['vol']     else "--"
        if e['barst']:
            barst_str = f"JA {e['barst_conf']:.2f}"
        elif e['intact_conf'] > 0:
            barst_str = f"nee {e['intact_conf']:.2f}"
        else:
            barst_str = "nee --"

        tekst = f"{e['i']+1:<3}{kleur_str:<7}{maat_str:<6}{breed_str:<7}{hoog_str:<6}{vol_str:<7}{barst_str}"
        cv2.putText(info_panel, tekst, (8, y), font, 0.96, (255, 255, 255), 2, cv2.LINE_AA)

    combined = np.hstack([display, info_panel])
    cv2.putText(combined, f"{len(egg_data)} eieren gedetecteerd", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Ei Analyse", combined)
    if cv2.waitKey(1) == 27:
        break

zed.close()
cv2.destroyAllWindows()
