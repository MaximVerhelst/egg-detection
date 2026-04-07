import cv2
import numpy as np
import os
import re
import csv

IMAGE_DIR = "test_instellingen"
EGG_ROI    = None  # (x, y, w, h) of None voor centrale crop
BG_ROI     = None  # (x, y, w, h) of None voor linkerbovenhoek
DEFECT_ROI = None  # (x, y, w, h) van het gebroken ei — verplicht te selecteren


def calculate_metrics(img, egg_roi, bg_roi, defect_roi):
    h, w = img.shape[:2]
    x, y, cw, ch = egg_roi or (int(w*.20), int(h*.20), int(w*.60), int(h*.60))
    bx, by, bw, bh = bg_roi or (10, 10, 100, 100)

    crop      = img[y:y+ch, x:x+cw]
    gray_full = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    sharpness  = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    brightness = float(gray_crop.mean())
    noise      = float(gray_full[by:by+bh, bx:bx+bw].std())
    total      = img.shape[0] * img.shape[1]
    over       = float(np.mean([cv2.calcHist([img],[c],None,[256],[0,256])[255][0]
                                for c in range(3)]) / total * 100)
    under      = float(np.mean([cv2.calcHist([img],[c],None,[256],[0,256])[0][0]
                                for c in range(3)]) / total * 100)

    # Defect visibility: lokaal randcontrast in de ROI van het gebroken ei
    if defect_roi is not None:
        dx, dy, dw, dh = defect_roi
        defect_crop = gray_full[dy:dy+dh, dx:dx+dw]
        defect_score = float(cv2.Laplacian(defect_crop, cv2.CV_64F).var())
    else:
        defect_score = sharpness  # fallback als geen ROI geselecteerd

    bp       = max(1.0 - abs(brightness - 128) / 128, 0.01)
    combined = defect_score * bp / (1.0 + over)

    return {
        'combined_score':  round(combined,      1),
        'defect_score':    round(defect_score,  1),
        'sharpness_score': round(sharpness,     1),
        'brightness':      round(brightness,    1),
        'noise':           round(noise,         2),
        'overexposed':     round(over,          2),
        'underexposed':    round(under,         2),
    }


def select_roi(frame, title, instruction, optional=False):
    preview = cv2.resize(frame, (1280, 720))
    print(instruction)
    if optional:
        print("  (Druk ENTER zonder te tekenen om over te slaan)")
    roi = cv2.selectROI(title, preview, fromCenter=False)
    cv2.destroyAllWindows()
    if roi == (0, 0, 0, 0):
        return None
    sx, sy = frame.shape[1] / 1280, frame.shape[0] / 720
    return (int(roi[0]*sx), int(roi[1]*sy), int(roi[2]*sx), int(roi[3]*sy))


def main():
    files = sorted([f for f in os.listdir(IMAGE_DIR)
                    if re.match(r'exp\d+_gain\d+_sh\d+_ct\d+\.png', f)])

    if not files:
        print(f"Geen foto's gevonden in '{IMAGE_DIR}'.")
        print("Voer eerst capture_settings.py uit.")
        return

    print(f"{len(files)} foto's gevonden in '{IMAGE_DIR}'.")

    # ROI selectie op eerste foto
    first_img = cv2.imread(os.path.join(IMAGE_DIR, files[0]))

    egg_roi = select_roi(first_img,
                         "Stap 1: Selecteer de EIEREN - druk ENTER",
                         "Teken een rechthoek rond alle eieren en druk ENTER.")

    bg_roi = select_roi(first_img,
                        "Stap 2: Selecteer donkere ACHTERGROND - druk ENTER",
                        "Teken een rechthoek op een donker leeg gebied en druk ENTER.",
                        optional=True)

    defect_roi = select_roi(first_img,
                            "Stap 3: Selecteer het GEBROKEN EI - druk ENTER",
                            "Teken een rechthoek ENKEL rond het gebroken ei en druk ENTER.\n"
                            "  --> Dit is de defect_score: hoe zichtbaar is de barst?",
                            optional=True)

    print(f"\nEier-ROI:        {egg_roi or 'centrale crop'}")
    print(f"Achtergrond-ROI: {bg_roi or 'linkerbovenhoek'}")
    print(f"Defect-ROI:      {defect_roi or 'niet geselecteerd (gebruikt algemene scherpte)'}")

    if defect_roi is None:
        print("\nWAARSCHUWING: Geen defect-ROI geselecteerd.")
        print("  De combined_score is gebaseerd op algemene scherpte.")
        print("  Voor maximale nauwkeurigheid: herstart en selecteer het gebroken ei.\n")
    else:
        print("\nDefect-ROI actief: combined_score = defect zichtbaarheid * belichtingskwaliteit\n")

    results = []
    for fname in files:
        m = re.match(r'exp(\d+)_gain(\d+)_sh(\d+)_ct(\d+)\.png', fname)
        img = cv2.imread(os.path.join(IMAGE_DIR, fname))
        if img is None:
            continue
        metrics = calculate_metrics(img, egg_roi, bg_roi, defect_roi)
        results.append({
            'filename': fname,
            'exposure': int(m[1]), 'gain': int(m[2]),
            'sharpness': int(m[3]), 'contrast': int(m[4]),
            **metrics
        })
        print(f"{fname}: Combined={metrics['combined_score']:.1f}  "
              f"Defect={metrics['defect_score']:.1f}  "
              f"Sharp={metrics['sharpness_score']:.1f}  "
              f"Bright={metrics['brightness']:.1f}  "
              f"Noise={metrics['noise']:.2f}  Over={metrics['overexposed']:.2f}%")

    results.sort(key=lambda r: r['combined_score'], reverse=True)

    # CSV opslaan
    csv_path = os.path.join(IMAGE_DIR, "resultaten.csv")
    fieldnames = ['filename', 'exposure', 'gain', 'sharpness', 'contrast',
                  'combined_score', 'defect_score', 'sharpness_score', 'brightness',
                  'noise', 'overexposed', 'underexposed']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV opgeslagen: {csv_path}")

    # Top 10
    print("\n=== TOP 10 (gesorteerd op combined_score = defect zichtbaarheid) ===")
    print(f"{'#':<4} {'Combined':<10} {'Defect':<9} {'Sharp':<9} {'Bright':<9} "
          f"{'Noise':<8} {'Over%':<7} {'Exp':<5} {'Gain':<6} {'Sh':<4} {'Ct'}")
    print("-" * 85)
    for i, r in enumerate(results[:10], 1):
        marker = " <-- BEST" if i == 1 else ""
        print(f"{i:<4} {r['combined_score']:<10} {r['defect_score']:<9} "
              f"{r['sharpness_score']:<9} {r['brightness']:<9} {r['noise']:<8} "
              f"{r['overexposed']:<7} {r['exposure']:<5} {r['gain']:<6} "
              f"{r['sharpness']:<4} {r['contrast']}{marker}")

    best = results[0]
    print(f"\nBeste combinatie voor defectdetectie:")
    print(f"  Exposure  = {best['exposure']}")
    print(f"  Gain      = {best['gain']}")
    print(f"  Sharpness = {best['sharpness']}")
    print(f"  Contrast  = {best['contrast']}")
    print(f"  Defect score = {best['defect_score']:.1f}  (hogere waarde = barst beter zichtbaar)")
    print(f"\nVoer nu export_rapport.py uit voor Excel en grafieken.")


if __name__ == "__main__":
    main()
