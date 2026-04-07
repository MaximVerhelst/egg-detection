from ultralytics import YOLO

EGG_YAML   = r"datasets\egg_detection\data.yaml"
BARST_YAML = r"datasets\crack_detection\data.yaml"
RUNS_DIR   = r"models"

# --- Egg detectie ---
print("=== EGG DETECTIE ===")
model_egg = YOLO("yolov8n.pt")
model_egg.train(
    data=EGG_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    name="egg_detection",
    project=RUNS_DIR,
    patience=15,
    seed=42,
)

# --- Crack detectie ---
print("\n=== CRACK DETECTION ===")
model_barst = YOLO("yolov8n.pt")
model_barst.train(
    data=BARST_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    name="crack_detection",
    project=RUNS_DIR,
    patience=15,
    seed=42,
)

print("\nTraining voltooid!")
print(f"Modellen opgeslagen in: {RUNS_DIR}")
