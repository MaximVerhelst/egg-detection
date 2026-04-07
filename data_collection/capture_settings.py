import pyzed.sl as sl
import cv2
import os
import time

# Testwaarden
EXPOSURES   = [15, 20, 25, 30, 35]
GAINS       = [20, 35, 50]
SHARPNESSES = [6, 8]
CONTRASTS   = [3, 5, 7]

SAVE_DIR = "test_instellingen"


def apply_setting(zed, setting, value):
    zed.set_camera_settings(setting, value)
    time.sleep(0.1)


def capture_frame(zed, runtime_params, image, retries=5):
    for _ in range(retries):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        time.sleep(0.2)
    return None


def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Camera kon niet openen")
        return

    # Vaste instellingen
    apply_setting(zed, sl.VIDEO_SETTINGS.SATURATION,               5)
    apply_setting(zed, sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO,        0)
    apply_setting(zed, sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, 4400)

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    os.makedirs(SAVE_DIR, exist_ok=True)

    total   = len(EXPOSURES) * len(GAINS) * len(SHARPNESSES) * len(CONTRASTS)
    current = 0

    print(f"Start: {total} combinaties worden gefotografeerd...")
    print(f"Foto's worden opgeslagen in '{SAVE_DIR}'\n")

    for exp in EXPOSURES:
        for gn in GAINS:
            for sh in SHARPNESSES:
                for ct in CONTRASTS:
                    current += 1
                    print(f"[{current}/{total}] Exp={exp} Gain={gn} Sharp={sh} Contrast={ct} ...", end=" ", flush=True)

                    apply_setting(zed, sl.VIDEO_SETTINGS.EXPOSURE,  exp)
                    apply_setting(zed, sl.VIDEO_SETTINGS.GAIN,      gn)
                    apply_setting(zed, sl.VIDEO_SETTINGS.SHARPNESS, sh)
                    apply_setting(zed, sl.VIDEO_SETTINGS.CONTRAST,  ct)
                    time.sleep(0.8)

                    frame = capture_frame(zed, runtime_params, image)
                    if frame is None:
                        print("MISLUKT")
                        continue

                    filename = f"exp{exp}_gain{gn}_sh{sh}_ct{ct}.png"
                    cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)
                    print("OK")

    zed.close()
    print(f"\nKlaar. {current} foto's opgeslagen in '{SAVE_DIR}'.")
    print("Voer nu 'calculate_scores.py' uit om de scores te berekenen.")


if __name__ == "__main__":
    main()