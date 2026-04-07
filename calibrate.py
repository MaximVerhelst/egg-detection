import pyzed.sl as sl
import cv2
import numpy as np
import json
import os

KALI_PATH = r"calibration.json"

EXPOSURE   = 15
GAIN       = 35
SHARPNESS  = 8
CONTRAST   = 7
SATURATION = 5
WHITBALANS = 4400

# --- Camera openen ---
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

# --- State ---
STATE_HOEKEN = 0
STATE_KARTON = 1
STATE_KLAAR  = 2

state         = STATE_HOEKEN
kali_punten   = []
plane_coeffs  = None
karton_offset = 0.0
current_depth = [None]
display_w_ref = {'w': 2208}


def get_median_depth(dmap, cx, cy, r=5):
    vals = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            _, v = dmap.get_value(cx + dx, cy + dy)
            if np.isfinite(v) and v > 0:
                vals.append(v)
    return float(np.median(vals)) if vals else None


def on_mouse(event, x, y, flags, param):
    global state, kali_punten, plane_coeffs, karton_offset

    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if x >= display_w_ref['w']:
        return

    dmap = current_depth[0]
    if dmap is None:
        return

    d = get_median_depth(dmap, x, y)
    if d is None:
        print(f"Geen geldige diepte op ({x}, {y})")
        return

    if state == STATE_HOEKEN:
        kali_punten.append((x, y, d))
        print(f"  Hoekpunt {len(kali_punten)}/4: pixel=({x},{y})  diepte={d:.1f}mm")
        if len(kali_punten) == 4:
            A = np.array([[u, v, 1] for u, v, _ in kali_punten], dtype=float)
            b_vec = np.array([dep for _, _, dep in kali_punten], dtype=float)
            coeffs, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            plane_coeffs = tuple(coeffs)
            print(f"\n  Vlak: d = {coeffs[0]:.6f}*u + {coeffs[1]:.6f}*v + {coeffs[2]:.1f}")
            print("\nFase 2: Leg een LEEG karton neer en klik op een lege plek")
            state = STATE_KARTON

    elif state == STATE_KARTON:
        plane_val = plane_coeffs[0] * x + plane_coeffs[1] * y + plane_coeffs[2]
        karton_offset = d - plane_val
        print(f"\n  Karton diepte: {d:.1f}mm  (offset: {karton_offset:+.1f}mm)")

        data = {
            'fx': fx,
            'fy': fy,
            'plane': list(plane_coeffs),
            'karton_offset': karton_offset,
        }
        with open(KALI_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n  Kalibratie opgeslagen: {KALI_PATH}")
        print("  Druk ESC om te sluiten.")
        state = STATE_KLAAR


cv2.namedWindow("Kalibratie", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Kalibratie", 1280, 720)
cv2.setMouseCallback("Kalibratie", on_mouse)

print("\n=== KALIBRATIE ===")
print("Fase 1: Klik 4 hoekpunten op het karton (zelfde hoogte, bv. randen karton)")

while True:
    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        continue

    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

    frame = image_zed.get_data()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    display = frame.copy()
    current_depth[0] = depth_map
    display_w_ref['w'] = display.shape[1]

    font = cv2.FONT_HERSHEY_SIMPLEX

    if state == STATE_HOEKEN:
        cv2.putText(display, f"Fase 1: Klik hoekpunt {len(kali_punten)+1}/4", (10, 45),
                    font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        for u, v, d in kali_punten:
            cv2.circle(display, (u, v), 12, (0, 255, 255), -1)
            cv2.putText(display, f"{d:.0f}mm", (u + 14, v - 6), font, 0.7, (0, 255, 255), 2)
        if len(kali_punten) >= 2:
            for j in range(len(kali_punten) - 1):
                cv2.line(display, kali_punten[j][:2], kali_punten[j+1][:2], (0, 255, 255), 1)

    elif state == STATE_KARTON:
        cv2.putText(display, "Fase 2: Klik lege plek op karton", (10, 45),
                    font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        for u, v, d in kali_punten:
            cv2.circle(display, (u, v), 12, (0, 255, 255), -1)

    elif state == STATE_KLAAR:
        cv2.putText(display, "Kalibratie opgeslagen! ESC om te sluiten.", (10, 45),
                    font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Kalibratie", display)
    if cv2.waitKey(1) == 27:
        break

zed.close()
cv2.destroyAllWindows()
