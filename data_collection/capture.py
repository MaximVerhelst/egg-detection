import pyzed.sl as sl
import cv2
import os

# Beste instellingen bepaald via calculate_scores.py
EXPOSURE   = 15
GAIN       = 35
SHARPNESS  = 8
CONTRAST   = 7
SATURATION = 5
WHITBALANS = 4400

SAVE_DIR = "images"

KLASSEN = {
    ord('1'): "ei",
    ord('2'): "geen_ei",
    ord('3'): "intact",
    ord('4'): "barst",
}


def count_existing(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".png")])


def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Camera kon niet openen")
        return

    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,                 EXPOSURE)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN,                     GAIN)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS,                SHARPNESS)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST,                 CONTRAST)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION,               SATURATION)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO,        0)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, WHITBALANS)

    for klasse in KLASSEN.values():
        os.makedirs(os.path.join(SAVE_DIR, klasse), exist_ok=True)

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    print("\n=== DATA VERZAMELEN ===")
    print("  1 = ei        2 = geen_ei")
    print("  3 = intact    4 = barst")
    print("  ESC = stoppen\n")
    for klasse in KLASSEN.values():
        print(f"  {klasse}: {count_existing(os.path.join(SAVE_DIR, klasse))} foto's")
    print()

    cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED", 1280, 720)

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            cv2.imshow("ZED", frame)
            key = cv2.waitKey(1)

            if key in KLASSEN:
                klasse = KLASSEN[key]
                folder = os.path.join(SAVE_DIR, klasse)
                n = count_existing(folder)
                filename = os.path.join(folder, f"{klasse}_{n:04d}.png")
                cv2.imwrite(filename, frame)
                score = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                print(f"[{klasse}] {filename}  |  Scherpte: {score:.1f}")

            elif key == 27:
                break

    print("\nAantal foto's per klasse:")
    for klasse in KLASSEN.values():
        print(f"  {klasse}: {count_existing(os.path.join(SAVE_DIR, klasse))}")

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()