import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"models\egg_detection\weights\best.pt"

EXPOSURE   = 15
GAIN       = 35
SHARPNESS  = 8
CONTRAST   = 7
SATURATION = 5
WHITBALANS = 4400

model = YOLO(MODEL_PATH)

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

cam_info = zed.get_camera_information()
fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
fy = cam_info.camera_configuration.calibration_parameters.left_cam.fy
print(f"Focale lengte: fx={fx:.1f}  fy={fy:.1f} px")

runtime_params = sl.RuntimeParameters()
image_zed = sl.Mat()
depth_map = sl.Mat()

# --- Kalibratie state ---
STATE_KALI_CAMERA = 0   # wacht op 4 hoekpunten
STATE_KALI_KARTON = 1   # wacht op 1 lege plek
STATE_METEN       = 2   # normaal meten

state        = STATE_KALI_CAMERA
kali_punten  = []        # (u, v, depth) voor plane fit
plane_coeffs = None      # (a, b, c): depth = a*u + b*v + c
karton_offset = 0.0      # offset na karton kalibratie
current_depth = [None]   # gedeeld met mouse callback


def get_median_depth(dmap, cx, cy, r=5):
    vals = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            _, v = dmap.get_value(cx + dx, cy + dy)
            if np.isfinite(v) and v > 0:
                vals.append(v)
    return float(np.median(vals)) if vals else None


def ref_depth(u, v):
    a, b, c = plane_coeffs
    return a * u + b * v + c + karton_offset


def on_mouse(event, x, y, flags, param):
    global state, kali_punten, plane_coeffs, karton_offset

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    display_w = param['display_w']
    if x >= display_w:
        return

    dmap = current_depth[0]
    if dmap is None:
        return

    d = get_median_depth(dmap, x, y)
    if d is None:
        print(f"Geen geldige diepte op ({x}, {y})")
        return

    if state == STATE_KALI_CAMERA:
        kali_punten.append((x, y, d))
        print(f"  Punt {len(kali_punten)}/4: pixel=({x},{y})  diepte={d:.1f}mm")
        if len(kali_punten) == 4:
            A = np.array([[u, v, 1] for u, v, _ in kali_punten], dtype=float)
            b_vec = np.array([dep for _, _, dep in kali_punten], dtype=float)
            coeffs, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            plane_coeffs = tuple(coeffs)
            print(f"  Plane fit: d = {coeffs[0]:.5f}*u + {coeffs[1]:.5f}*v + {coeffs[2]:.1f}")
            print("\nFase 2: Klik op een LEGE plek op het karton")
            state = STATE_KALI_KARTON

    elif state == STATE_KALI_KARTON:
        plane_val = plane_coeffs[0] * x + plane_coeffs[1] * y + plane_coeffs[2]
        karton_offset = d - plane_val
        print(f"  Karton diepte: {d:.1f}mm  (offset: {karton_offset:+.1f}mm)")
        print("\nKalibratie voltooid! Meten gestart. ESC = stoppen.")
        state = STATE_METEN


cv2.namedWindow("Afmetingen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Afmetingen", 1600, 720)

display_w_ref = {'display_w': 2208}
cv2.setMouseCallback("Afmetingen", on_mouse, display_w_ref)

print("\nFase 1: Klik 4 hoekpunten op het karton (zelfde hoogte)")

while True:
    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        continue

    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

    frame = image_zed.get_data()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    display = frame.copy()
    current_depth[0] = depth_map
    display_w_ref['display_w'] = display.shape[1]

    info_h = display.shape[0]
    info_w = 320
    font = cv2.FONT_HERSHEY_SIMPLEX
    info_panel = np.zeros((info_h, info_w, 3), dtype=np.uint8)

    if state == STATE_KALI_CAMERA:
        msg = f"Fase 1: Klik hoekpunt {len(kali_punten)+1}/4"
        cv2.putText(display, msg, (10, 45), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        for u, v, d in kali_punten:
            cv2.circle(display, (u, v), 10, (0, 255, 255), -1)
            cv2.putText(display, f"{d:.0f}mm", (u + 12, v - 5), font, 0.65, (0, 255, 255), 2)

    elif state == STATE_KALI_KARTON:
        cv2.putText(display, "Fase 2: Klik lege plek op karton", (10, 45), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        for u, v, d in kali_punten:
            cv2.circle(display, (u, v), 10, (0, 255, 255), -1)

    elif state == STATE_METEN:
        results = model(frame, conf=0.3, verbose=False)[0]
        egg_data = []

        boxes_sorted = sorted(results.boxes, key=lambda b: (int(b.xyxy[0][1]) // 100, int(b.xyxy[0][0])))

        for i, box in enumerate(boxes_sorted):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx, ry = (x2 - x1) // 2, (y2 - y1) // 2

            d_ei = get_median_depth(depth_map, cx, cy)

            if d_ei is None:
                egg_data.append({'i': i, 'breedte': None, 'hoogte': None, 'vol': None})
                cv2.ellipse(display, (cx, cy), (rx, ry), 0, 0, 360, (0, 0, 255), 2)
                cv2.putText(display, f"#{i+1}", (cx - 20, cy + 10), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                continue

            d_karton = ref_depth(cx, cy)
            hoogte_ei = d_karton - d_ei         # hoogte van het ei in mm
            b_px = min(rx, ry)                  # korte as in pixels
            breedte_mm = (b_px * 2) * d_ei / fx # breedte via korte as

            a = hoogte_ei / 2   # halve lange as
            b = breedte_mm / 2  # halve korte as

            if a > 0 and b > 0:
                vol_cm3 = (4/3) * np.pi * a * b**2 / 1000
            else:
                vol_cm3 = None

            egg_data.append({'i': i, 'breedte': breedte_mm, 'hoogte': hoogte_ei, 'vol': vol_cm3})

            cv2.ellipse(display, (cx, cy), (rx, ry), 0, 0, 360, (0, 255, 0), 2)
            # Teken b-as (korte as) in cyaan
            if rx <= ry:
                cv2.line(display, (cx - b_px, cy), (cx + b_px, cy), (255, 255, 0), 2)
            else:
                cv2.line(display, (cx, cy - b_px), (cx, cy + b_px), (255, 255, 0), 2)
            cv2.putText(display, f"#{i+1}", (cx - 20, cy + 10), font, 1.0, (255, 100, 0), 2, cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 4, (0, 255, 255), -1)

        # Info paneel
        cv2.putText(info_panel, "#   Breed  Hoog   Vol(cm3)", (8, 28), font, 0.55, (255, 255, 255), 2)
        cv2.line(info_panel, (5, 36), (info_w - 5, 36), (80, 80, 80), 1)

        for r in egg_data:
            y = 56 + r['i'] * 26
            if y > info_h - 30:
                break
            if r['breedte'] is None:
                cv2.putText(info_panel, f"{r['i']+1:<3}  geen diepte", (8, y), font, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
            elif r['vol'] is None:
                tekst = f"{r['i']+1:<3}  {r['breedte']:.0f}mm  {r['hoogte']:.0f}mm  --"
                cv2.putText(info_panel, tekst, (8, y), font, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
            else:
                tekst = f"{r['i']+1:<3}  {r['breedte']:.0f}mm  {r['hoogte']:.0f}mm  {r['vol']:.1f}"
                cv2.putText(info_panel, tekst, (8, y), font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        n = len(egg_data)
        cv2.putText(info_panel, f"{n} eieren", (8, info_h - 10), font, 0.55, (200, 200, 200), 1)

    combined = np.hstack([display, info_panel])
    cv2.imshow("Afmetingen", combined)
    if cv2.waitKey(1) == 27:
        break

zed.close()
cv2.destroyAllWindows()
