import random
import glob
import os
import numpy as np
from PIL import Image

# PyTorch imports
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset



class HemoSetDataset(Dataset):

    def __init__(
        self,
        data_dir,
        history_length=4,
        augment=True,
        img_size=(80, 80),
        skip_rate=0
    ):
        """
        Initializes the HemoSet Dataset from a given path to HemoSet's folder. 

        :param str data_dir: Root directory of the multilabel dataset folder. 
                             Both image and mask are in .png format.
        :param int history_length: The number of history frames included as 
                                   reference. Default 4 frames. 
        :param bool augment: Whether or not to strengthen the dataset by
                             performing random manipulations on the image. 
                             Default True. 
        :param tuple img_size: Image output size (H in pixels, W in pixels)
                               Default (80, 80)
        :param int skip_rate: How many frames to skip between sequences.
                              skip_rate=0 gives consecutive sequences (1234, 2345, 3456)
                              skip_rate=1 gives sequences spaced by 1 (1357, 2468, 3579)
                              Default 0.
        """
        self.data_dir = data_dir
        self.history_length = history_length
        self.augment = augment
        self.img_size = img_size
        self.skip_rate = skip_rate
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

        self.video_frames = self._organize_files()
        self.sequences = self._generate_sequences()

    def _organize_files(self):
        """
        Organize input files (image and mask) based on the structure of HemoSet 

        HemoSet folder structure:
        HemoSet/
        ├── pig1/
        │   ├── imgs/
        │   │   ├── 000000.png
        │   │   ├── 000030.png
        │   │   └── ...
        │   └── labels/
        │       ├── 000000_mask.png
        │       ├── 000030_mask.png
        │       └── ...
        ├── pig2/
        │   ├── imgs/
        │   └── labels/
        └── ...

        :return: Dict with key as pig_id, value as a list of frame info
                 Every frame_info is a dict with the following information:
                 {
                    'image_path': 'path/to/XXXXXX.png',
                    'mask_path': 'path/to/XXXXXX_mask.png',
                    'pig_id': pig_id
                 }
        """

        video_dict = {}

        pig_folders = glob.glob(os.path.join(self.data_dir, "pig*"))
        pig_folders = [f for f in pig_folders if os.path.isdir(f)]

        for pig_folder in pig_folders:
            pig_id = int(os.path.basename(pig_folder)[3:])

            img_folder = os.path.join(pig_folder, "imgs")
            label_folder = os.path.join(pig_folder, "labels")

            # Load images as a list of strings, then sort by frame number
            image_files = glob.glob(os.path.join(img_folder, "*.png"))
            image_files = [f for f in image_files]
            image_files.sort(
                key=lambda x: int(os.path.basename(x).split(".")[0])
            )
            
            # Collect masks for each image
            frame_list = []
            for image_file in image_files:
                frame_num = os.path.basename(image_file).split(".")[0]
                mask_file = os.path.join(label_folder, f"{frame_num}_mask.png")

                frame_info = {
                    "image_path": image_file,
                    "mask_path": mask_file,
                    "pig_id": pig_id
                }
                frame_list.append(frame_info)

            video_dict[pig_id] = frame_list

        return video_dict

    def _generate_sequences(self):
        sequences = []
        for pig_id, frame_info in self.video_frames.items():
            seq_span = self.history_length * (self.skip_rate + 1)
            
            if len(frame_info) < seq_span:
                continue

            for start_idx in range(0, len(frame_info) - seq_span + 1):
                sequence = frame_info[start_idx : start_idx + seq_span : (self.skip_rate + 1)]
                assert len(sequence) == self.history_length, \
                    f"Sequence length incorrect. Expected: {self.history_length}, actual: {len(sequence)}"
                sequences.append(sequence)
        return sequences

    def _apply_augmentation(
        self,
        frames,
        masks,
        augmentation_type
    ):
        """
        Apply augmentation to frames and masks.
        """
        if augmentation_type == "rotate":
            angle = random.choice([90, 180, 270])
            frames = [TF.rotate(frame, angle) for frame in frames]
            masks = [TF.rotate(mask, angle) for mask in masks]
            masks = [self._recalculate_background(mask) for mask in masks]
        elif augmentation_type == "brightness_contrast":
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.85, 1.15)
            frames = [TF.adjust_brightness(frame, brightness_factor) for frame in frames]
            frames = [TF.adjust_contrast(frame, contrast_factor) for frame in frames]
        elif augmentation_type == "gaussian_noise":
            frames = [frame + torch.randn_like(frame) * 0.025 for frame in frames]
        return frames, masks

    def _recalculate_background(self, mask):
        """
        Making sure background still accurate after rotation.
        Risk comes from resizing.
        """
        mask = mask.clone()
        organs_sum = torch.sum(mask[:7], dim=0)
        mask[7] = 1.0 - torch.clamp(organs_sum, 0, 1)
        return mask

    def _load_masks(self, mask_path):
        """
        Load a single binary mask
        #000000 (background) and #800000 (blood)
        """
        mask = np.array(Image.open(mask_path).convert("RGB"))
    
        blood_mask = (mask[..., 0] > 110) & (mask[..., 1] == 0) & \
                 (mask[..., 2] == 0)
        full_mask = np.zeros((2, *mask.shape[:2]), dtype=np.float32)
        full_mask[0] = blood_mask.astype(np.float32)
        full_mask[1] = 1.0 - full_mask[0]

        return torch.from_numpy(full_mask)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(
        self,
        idx
    ):
        """
        Get a sequence (frames with length history_length) by index. 
        Returns frames and masks
        """
        sequence_info = self.sequences[idx]
        frames = torch.zeros((self.history_length, 3, *self.img_size))
        masks = torch.zeros((self.history_length, 2, *self.img_size))


        augmentation_type = None

        if isinstance(sequence_info, tuple):
            sequence, augmentation_type = sequence_info
        else:
            sequence = sequence_info

        frames = []
        masks = []

        for frame_info in sequence:
            img = Image.open(frame_info["image_path"]).convert("RGB")
            img = self.transform(img).clone().detach()

            mask = self._load_masks(frame_info["mask_path"])
            mask = TF.resize(mask, self.img_size).clone().detach()

            frames.append(img)
            masks.append(mask)
        
        if augmentation_type:
            frames, masks = self._apply_augmentation(
                frames,
                masks,
                augmentation_type
            )

        assert masks[0].shape[0] == 2, f"Mask should be 2-channel, got {masks[0].shape}"
        frames = torch.stack(frames, dim=0)  # shape: [history_length, C, H, W]
        masks = torch.stack(masks, dim=0)    # shape: [history_length, num_classes, H, W]

        return frames, masks