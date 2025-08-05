import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from DMBMSNet import DMBMSNet
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))

num_classes = 13
num_encoder_blocks = 5
num_heads = 8
k = 5
memory_length = 4
num_memory_blocks = 4
deformable = True
use_crf = True

model = DMBMSNet(
    input_dim=3,
    num_classes=num_classes,
    num_encoder_blocks=num_encoder_blocks,
    base_dim=64,
    num_heads=num_heads,
    k=k,
    memory_frames=memory_length,
    num_memory_blocks=num_memory_blocks,
    deformable=deformable,
    use_crf=use_crf
).to(device)

checkpoint_path = "D:/College/Research/SLU/laparoscopic surgery/LaparoSeg/best_model_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

CLASS_COLORS = {
    0: (127, 127, 127),
    1: (210, 140, 140),
    2: (255, 114, 114),
    3: (231, 70, 156), 
    4: (186, 183, 75),
    5: (170, 255, 0), 
    6: (255, 85, 0),   
    7: (255, 0, 0),    
    8: (255, 255, 0),  
    9: (169, 255, 184),
    10: (255, 160, 165),  
    11: (0, 50, 128),  
    12: (111, 74, 0) 
}

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)

def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((80, 80), Image.NEAREST)
    mask_array = np.array(mask)
    
    full_mask = np.zeros((num_classes, *mask_array.shape), dtype=np.float32)
    
    for class_id in range(num_classes):
        full_mask[class_id] = (mask_array == class_id).astype(np.float32)
    
    return torch.from_numpy(full_mask)

def get_sequence_frames(base_path, frame_num, history_length=4):
    dir_path = os.path.dirname(base_path)
    base_name = os.path.basename(base_path).split('_')[0]
    
    all_frames = sorted(glob.glob(os.path.join(dir_path, f"{base_name}_*.png")))
    
    frame_numbers = []
    for f in all_frames:
        try:
            num = int(os.path.basename(f).split('_')[1])
            frame_numbers.append(num)
        except:
            continue
    
    current_idx = frame_numbers.index(frame_num)
    start_idx = max(0, current_idx - history_length)
    
    return [all_frames[i] for i in range(start_idx, current_idx + 1)]

def visualize_results(img_path, pre_crf_pred, post_crf_pred, mask_path=None):
    plt.figure(figsize=(24, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(Image.open(img_path))
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    pred_rgb = np.zeros((*pre_crf_pred.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        pred_rgb[pre_crf_pred == class_id] = color
    plt.imshow(pred_rgb)
    plt.title('Pre-CRF Prediction')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    pred_rgb = np.zeros((*post_crf_pred.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        pred_rgb[post_crf_pred == class_id] = color
    plt.imshow(pred_rgb)
    plt.title('Post-CRF Prediction')
    plt.axis('off')
    
    if mask_path:
        color_mask_path = mask_path.replace("_endo_watershed_mask.png", "_endo_color_mask.png")
        plt.subplot(1, 4, 4)
        color_mask = Image.open(color_mask_path).convert("RGB")
        color_mask = color_mask.resize((80, 80), Image.NEAREST)
        plt.imshow(color_mask)
        plt.title('Ground Truth')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{os.path.basename(img_path).split('.')[0]}_result.png")
    plt.show()

def test_single_image(current_frame_path, mask_path=None): 
    current_frame = preprocess_image(current_frame_path)
    
    with torch.no_grad():
        pre_crf_output, post_crf_output = model(
            current_frame,
            training=False
        )
    
    pre_crf_pred = torch.argmax(pre_crf_output.squeeze(), dim=0).cpu().numpy()
    post_crf_pred = torch.argmax(post_crf_output.squeeze(), dim=0).cpu().numpy()
    
    visualize_results(current_frame_path, pre_crf_pred, post_crf_pred, mask_path)

def test_sequence(current_frame_path, mask_path=None):
    frame_num = int(os.path.basename(current_frame_path).split('_')[1])
    
    frame_paths = get_sequence_frames(current_frame_path, frame_num, memory_length)
    
    frames = [preprocess_image(fp) for fp in frame_paths]
    current_frame = frames[-1] 
    memory_frames = torch.cat(frames[:-1], dim=0) 
    
    if mask_path:
        mask_dir = os.path.dirname(mask_path)
        mask_paths = [fp.replace("_endo.png", "_endo_watershed_mask.png") for fp in frame_paths]
        masks = [load_mask(mp) for mp in mask_paths]
        memory_masks = torch.stack(masks[:-1], dim=0).to(device)
    else:
        memory_masks = torch.zeros((memory_length, num_classes, 80, 80)).to(device)
    
    with torch.no_grad():
        pre_crf_output, post_crf_output = model(
            current_frame,
            training=False,
            memory_frames=memory_frames,
            memory_masks=memory_masks
        )
    
    pre_crf_pred = torch.argmax(pre_crf_output.squeeze(), dim=0).cpu().numpy()
    post_crf_pred = torch.argmax(post_crf_output.squeeze(), dim=0).cpu().numpy()
    
    visualize_results(current_frame_path, pre_crf_pred, post_crf_pred, mask_path)

test_images = [
    ("D:/College/Research/SLU/laparoscopic surgery/Dataset/CholecSeg8k/video52/video52_00240/frame_253_endo.png",
     "D:/College/Research/SLU/laparoscopic surgery/Dataset/CholecSeg8k/video52/video52_00240/frame_253_endo_watershed_mask.png"),
    ("D:/College/Research/SLU/laparoscopic surgery/Dataset/CholecSeg8k/video48/video48_00641/frame_646_endo.png",
     "D:/College/Research/SLU/laparoscopic surgery/Dataset/CholecSeg8k/video48/video48_00641/frame_646_endo_watershed_mask.png"),
    ("D:/College/Research/SLU/laparoscopic surgery/Dataset/CholecSeg8k/video55/video55_00508/frame_522_endo.png",
     "D:/College/Research/SLU/laparoscopic surgery/Dataset/CholecSeg8k/video55/video55_00508/frame_522_endo_watershed_mask.png")
]

for img_path, mask_path in test_images:
    print(f"\nTesting sequence starting with: {img_path}")
    if (memory_length == 0):
        test_single_image(img_path, mask_path)
    else:
        test_sequence(img_path, mask_path)
