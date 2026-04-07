import pyzed.sl as sl
import cv2
import os

# Startwaarden (gebaseerd op beste instellingen uit test_auto.py)
SAVE_DIR = "images"

def apply_postprocessing(frame, denoise, sharpen, clahe_obj, use_clahe):
    if denoise:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)
    if sharpen:
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        frame = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
    if use_clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Camera kon niet openen")
        return

    exposure   = 30
    gain       = 20
    sharpness  = 8
    contrast   = 3
    saturation = 5
    wb         = 4400

    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,                 exposure)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN,                     gain)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS,                sharpness)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST,                 contrast)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION,               saturation)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO,        0)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, wb)

    denoise   = False
    sharpen   = False
    use_clahe = True
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    os.makedirs(SAVE_DIR, exist_ok=True)
    existing = [f for f in os.listdir(SAVE_DIR) if f.startswith("frame_") and f.endswith(".png")]
    frame_id = len(existing)

    print("=== Toetsenbord overzicht ===")
    print("Exposure:    e=lager   r=hoger")
    print("Gain:        d=lager   f=hoger")
    print("Sharpness:   t=lager   y=hoger")
    print("Contrast:    g=lager   h=hoger")
    print("Saturation:  b=lager   n=hoger")
    print("Witbalans:   v=lager   m=hoger  (stappen van 100K)")
    print("Denoise:     1=aan/uit")
    print("Verscherpen: 2=aan/uit")
    print("CLAHE:       3=aan/uit")
    print("Instellingen tonen: i")
    print("Opslaan:     s  |  Stoppen: ESC")

    cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED", 1280, 720)

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = apply_postprocessing(frame, denoise, sharpen, clahe_obj, use_clahe)

            cv2.imshow("ZED", frame)
            key = cv2.waitKey(1)

            if key == ord('e') and exposure > 0:
                exposure -= 5
                zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
                print(f"Exposure: {exposure}")
            elif key == ord('r') and exposure < 100:
                exposure += 5
                zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
                print(f"Exposure: {exposure}")
            elif key == ord('d') and gain > 0:
                gain -= 5
                zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)
                print(f"Gain: {gain}")
            elif key == ord('f') and gain < 100:
                gain += 5
                zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)
                print(f"Gain: {gain}")
            elif key == ord('t') and sharpness > 0:
                sharpness -= 1
                zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, sharpness)
                print(f"Sharpness: {sharpness}")
            elif key == ord('y') and sharpness < 8:
                sharpness += 1
                zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, sharpness)
                print(f"Sharpness: {sharpness}")
            elif key == ord('g') and contrast > 0:
                contrast -= 1
                zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, contrast)
                print(f"Contrast: {contrast}")
            elif key == ord('h') and contrast < 8:
                contrast += 1
                zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, contrast)
                print(f"Contrast: {contrast}")
            elif key == ord('b') and saturation > 0:
                saturation -= 1
                zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, saturation)
                print(f"Saturation: {saturation}")
            elif key == ord('n') and saturation < 8:
                saturation += 1
                zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, saturation)
                print(f"Saturation: {saturation}")
            elif key == ord('v') and wb > 2800:
                wb -= 100
                zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, wb)
                print(f"Witbalans: {wb}K")
            elif key == ord('m') and wb < 6500:
                wb += 100
                zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, wb)
                print(f"Witbalans: {wb}K")
            elif key == ord('1'):
                denoise = not denoise
                print(f"Denoise: {'aan' if denoise else 'uit'}")
            elif key == ord('2'):
                sharpen = not sharpen
                print(f"Verscherpen: {'aan' if sharpen else 'uit'}")
            elif key == ord('3'):
                use_clahe = not use_clahe
                print(f"CLAHE: {'aan' if use_clahe else 'uit'}")
            elif key == ord('i'):
                print(f"\n--- Instellingen ---")
                print(f"Exposure={exposure}  Gain={gain}  Sharpness={sharpness}  Contrast={contrast}")
                print(f"Saturation={saturation}  Witbalans={wb}K")
                print(f"Denoise={'aan' if denoise else 'uit'}  Sharpen={'aan' if sharpen else 'uit'}  CLAHE={'aan' if use_clahe else 'uit'}")
            elif key == ord('s'):
                filename = os.path.join(SAVE_DIR, f"frame_{frame_id:04d}.png")
                cv2.imwrite(filename, frame)
                score = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                print(f"Opgeslagen: {filename}  |  Scherpte: {score:.1f}")
                frame_id += 1
            elif key == 27:
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()