import os
import random

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
from PIL import Image
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
import pdb
import math
import torchvision.transforms as T

cv2.setNumThreads(0)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """

    def __init__(self,
            data_path,
            filenames,
            height,
            width,
            frame_idxs,
            num_scales,
            is_train=False,
            robust_val=False,
            img_ext='.jpg',
            mask_noise=False,
            feat_warp=False,
            vertical=False,
            tiling=False,
            foggy=False,
            stereo_split=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.nuscenes_data = "/media/kieran/SSDNEW/Base-Model/data/nuScenes_RAW"
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.sensor = 'CAM_FRONT'
        self.stereo_split = stereo_split

        self.interp = T.InterpolationMode.LANCZOS

        self.foggy = foggy

        self.frame_idxs = frame_idxs
        self.robust_val = robust_val

        self.is_train = is_train
        self.img_ext = img_ext

        self.vertical = vertical
        self.tiling = tiling

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.mask_noise = mask_noise
        self.feat_warp = feat_warp

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                interpolation=self.interp)

    def tile_crop(self, color_aug_f, do_tiling, selection, order):
        if do_tiling:
            _, h, w = color_aug_f.shape
            height_selection = selection[0]
            width_selection = selection[1]
            height_split = h // height_selection
            width_split = w // width_selection
            selection_prod = np.prod(selection)
            # has to be divisabel by 3 and 4
            tiles = [color_aug_f[:, x:x + height_split, y:y + width_split] for x in range(0, h, height_split) for y in range(0, w, width_split)]
            tiles = [tiles[i] for i in order]
            width_cat = [torch.cat(tiles[i:width_selection + i], dim=2) for i in range(0, selection_prod, width_selection)]
            final = torch.cat(width_cat, dim=1)
        else:
            final = color_aug_f
        return final

    def vertical_crop(self, color_aug_f, do_vertical, rand_w):
        '''Applies a vertical dependence augmentation
        '''
        if do_vertical and rand_w > 0:
            output_image = []
            in_h = color_aug_f.shape[1]
            cropped_y = [0, int(rand_w * in_h), in_h]
            cropped_image = [color_aug_f[:, cropped_y[n]:cropped_y[n + 1], :] for n in range(2)]
            a = cropped_image[::-1]
            output_image = torch.cat(a, dim=1)
        else:
            output_image = color_aug_f
        return output_image

    def preprocess(self, inputs, color_aug, erase_aug, do_vertical, do_scale, small, height_re, width_re, box_HiS, do_flip, order, do_tiling, selection, rand_w, spec):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if ("color" in k):
                n, im, i = k
                for i in range(self.num_scales):
                    if n == "color":
                        inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])  # n = color
                        inputs[("scale_aug", im, i)] = inputs[(n, im, i)]  # n = color

        for k in list(inputs):
            f = inputs[k]
            if ("color" in k)  or ("scale_aug" in k):
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)

    def __len__(self):
        return len(self.filenames)

    def load_intrinsics_kitti(self, folder, frame_index):
        return self.K.copy()

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        spec = 'data'
        do_color_aug = False
        do_vertical = False
        do_tiling = False
        do_scale = False
        small = False
        do_flip = False
        rand_erase = False
        geometric = ''
        height_re = 0
        width_re = 0
        dx = 0
        dy = 0
        box_HiS = 0
        for i in range(self.num_scales):
            inputs[("dxy", i)] = torch.Tensor((0, 0))
            inputs[("resize", i)] = torch.Tensor((0, 0))

        poses = {}
        if type(self).__name__ == "CityscapesDataset":
            folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
            inputs.update(self.get_colors(folder, frame_index, side, do_flip, 'data', augs=False, foggy=self.foggy))
            if self.is_train or self.robust_val:
                inputs.update(self.get_colors(folder, frame_index, side, do_flip, spec, augs=True))
            inputs["dataset"] = 1

            for scale in range(self.num_scales):
                K = self.load_intrinsics(folder, frame_index)
                if do_scale:
                    K[0, :] *= width_re // (2 ** scale)
                    K[1, :] *= height_re // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
                else:
                    K[0, :] *= self.width // (2 ** scale)
                    K[1, :] *= self.height // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        elif type(self).__name__ == "KITTIRAWDataset" or type(self).__name__ == "KITTIOdomDataset":
            inputs["dataset"] = 0
            if self.is_robust_test:
                folder, frame_index, side, spec = self.index_to_folder_and_frame_idx(index)
                if self.robust_augment != None:
                    spec = self.robust_augment
            else:
                folder, frame_index, side, _ = self.index_to_folder_and_frame_idx(index)
            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(
                        folder, frame_index, other_side, "data", do_flip)
                else:
                    try:
                        inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, "data", do_flip)
                    except FileNotFoundError as e:
                        if i != 0:
                            # fill with dummy values
                            inputs[("color", i, -1)] = Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                            poses[i] = None
                        else:
                            raise FileNotFoundError(f'Cannot find frame - make sure your '
                                                    f'--data_path is set correctly, or try adding'
                                                    f' the --png flag. {e}')
            for scale in range(self.num_scales):
                K = self.load_intrinsics_kitti(folder, frame_index)
                if do_scale:
                    K[0, :] *= width_re // (2 ** scale)
                    K[1, :] *= height_re // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
                else:
                    K[0, :] *= self.width // (2 ** scale)
                    K[1, :] *= self.height // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        elif type(self).__name__ == "NuScenesDataset":
            inputs["dataset"] = 2
            new_index = self.get_correct_index(index)
            sample = self.get_sample_data(new_index)
            for i in self.frame_idxs:
                if i == "s":
                    raise NotImplementedError('nuscenes dataset does not support stereo depth')
                else:
                    inputs[("color", i, -1)] = self.get_color_nuscenes(sample, i, do_flip)

            for scale in range(self.num_scales):
                K = self.load_intrinsics_nuscenes(sample)
                if do_scale:
                    K[0, :] *= width_re // (2 ** scale)
                    K[1, :] *= height_re // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
                else:
                    K[0, :] *= self.width // (2 ** scale)
                    K[1, :] *= self.height // (2 ** scale)
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", scale)] = torch.from_numpy(K)
                    inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        elif type(self).__name__ == "DRIVINGSTEREO":

            inputs[("color", 0, -1)] = self.get_color(self.filenames[index], self.stereo_split)

        elif type(self).__name__ == "NUSCENESEVAL":

            new_index = self.get_correct_index(index)

            inputs[("color", 0, -1)] = self.get_color(new_index)

        color_aug = (lambda x: x)
        erase_aug = (lambda x: x)
        rand_w = 0
        selection = (0, 0)
        order = [0] * 12

        self.preprocess(inputs, color_aug, erase_aug, do_vertical, do_scale, small, height_re, width_re, box_HiS, do_flip, order, do_tiling, selection, rand_w, spec)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color", i, 1)]
            del inputs[("color", i, 2)]
            del inputs[("color", i, 3)]

        inputs["index"] = index
        inputs["rand_w"] = rand_w
        inputs["order"] = torch.tensor(order)
        inputs["do_tiling"] = do_tiling
        inputs["tile_selection"] = torch.tensor(selection)

        new_dict = {}
        inputs["distribution"] = new_dict

        inputs["do_scale"] = do_scale
        inputs["small"] = small

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
