import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO
from tqdm import tqdm

def apply_bad_composite_effect(frame, mask):
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    h, w = frame.shape[:2]

    erosion_mask = (np.random.rand(h, w) > 0.10).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, erosion_mask)

    actor = cv2.bitwise_and(frame, frame, mask=mask)
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    halo_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    halo_edge = cv2.subtract(halo_mask, mask)
    blurred_halo = cv2.GaussianBlur(halo_edge, (11, 11), sigmaX=5)

    green_halo = np.zeros_like(frame, dtype=np.float32)
    green_halo[:, :, 1] = blurred_halo.astype(np.float32) * 0.1
    actor = actor.astype(np.float32) + green_halo

    outline = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    edge = cv2.subtract(outline, mask)
    matte_edge = np.zeros_like(actor)
    matte_edge[edge > 0] = (0, 0, 0)
    actor += matte_edge.astype(np.float32)

    gradient = np.tile(np.linspace(1.7, 0.4, h).reshape(h, 1), (1, w))
    gradient = np.repeat(gradient[:, :, np.newaxis], 3, axis=2)
    actor *= gradient

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - w // 2)**2 + (Y - h // 2)**2)
    radial = 1.0 - (dist / np.max(dist))
    spotlight = np.clip(radial * 2.0, 0.6, 1.8)
    spotlight = np.repeat(spotlight[:, :, np.newaxis], 3, axis=2)
    actor *= spotlight

    avg_bg_color = cv2.mean(background, mask=cv2.bitwise_not(mask))[:3]
    r, g, b = avg_bg_color[2], avg_bg_color[1], avg_bg_color[0]
    warmth_score = r - b
    if warmth_score > 5:
        actor[:, :, 0] *= 1.5  
        actor[:, :, 2] *= 0.65 
    elif warmth_score < -10:
        actor[:, :, 0] *= 0.65
        actor[:, :, 2] *= 1.5

    actor_final = np.clip(actor, 0, 255).astype(np.uint8)
    composite = cv2.add(actor_final, background)
    return composite

def generate_bad_composite(input_path, output_path, model):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[!] Skipping unreadable image: {input_path}")
        return

    results = model(frame)
    masks = results[0].masks.data.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    for i, cls_id in enumerate(class_ids):
        if int(cls_id) == 0:
            mask = (masks[i] * 255).astype(np.uint8)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            frame = apply_bad_composite_effect(frame, mask)

    cv2.imwrite(output_path, frame)

if __name__ == "__main__":
    import logging
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Batch generate bad composites.")
    parser.add_argument("--input_dir", required=True, help="Input directory with movie frames")
    parser.add_argument("--output_dir", required=True, help="Output directory to save composites")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = YOLO("yolov8n-seg.pt")
    model.verbose = False

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Processing {len(image_files)} frames...")
    progress = tqdm(image_files, desc="Generating Bad Composites")
    for filename in progress:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        generate_bad_composite(input_path, output_path, model)

