import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from supervision.draw.color import Color, ColorPalette
from PIL import Image, ImageDraw

def visualize_parsing(parsing_map, img, alpha=0.5):
    colormap_bgr = {
        0: (0, 0, 0),            
        1: (189, 224, 255),      
        2: (0, 0, 255),          
        3: (0, 85, 255),         
        4: (0, 170, 255),        
        5: (0, 255, 255),        
        6: (0, 255, 170),        
        7: (0, 255, 85),         
        8: (0, 255, 0),          
        9: (85, 255, 0),         
        10: (255, 255, 0),      
        11: (255, 170, 0),       
        12: (255, 85, 0),        
        13: (255, 0, 0),         
        14: (255, 0, 85),        
        15: (255, 0, 170),       
        16: (255, 0, 255),       
        17: (170, 0, 255),       
        18: (85, 0, 255)    
    }

    label_names = {
        0: "background",
        1: "skin",
        2: "left brow",
        3: "right brow",
        4: "left eye",
        5: "right eye",
        6: "eye_g",      
        7: "left ear",
        8: "right ear",
        9: "ear ring",
        10: "nose",
        11: "mouth",
        12: "upper lip",
        13: "lower lip",
        14: "neck",
        15: "neck border",
        16: "cloth",
        17: "hair",
        18: "hat"
    }

    h, w = parsing_map.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in colormap_bgr.items():
        color_mask[parsing_map == label] = color

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    blended = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

    colormap_rgb = {k: (v[2]/255.0, v[1]/255.0, v[0]/255.0) for k, v in colormap_bgr.items()}
    unique_labels = np.unique(parsing_map)
    legend_patches = [
        Patch(color=colormap_rgb[lbl], label=label_names[lbl])
        for lbl in unique_labels if lbl in label_names
    ]

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title("Face Segmentation Map")
    plt.axis("off")
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualize_face_detections(image_path, detections):
    manual_colors = [
        Color(255, 0, 0), Color(0, 255, 0), Color(0, 0, 255), Color(255, 255, 0),
        Color(255, 0, 255), Color(0, 255, 255), Color(255, 128, 0), Color(128, 0, 255),
        Color(0, 128, 255), Color(128, 128, 128)
    ]
    palette = ColorPalette(colors=manual_colors)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for i, (box, conf, _) in enumerate(detections):
        x1, y1, x2, y2 = box
        label = f"Face {i+1} ({conf:.2f})"
        color_obj = palette.by_idx(i)
        color = (color_obj.r, color_obj.g, color_obj.b)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 10), label, fill=color)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Detected faces")
    plt.show()
