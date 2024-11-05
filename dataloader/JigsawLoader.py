import os
from random import random
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils import mask_utils


def load_pretraining_dataset(dataset, val_size):
    '''

    :param dataset: path to the csv file
    :param val_size: [0, 1]
    :return: train_df, val_df
    '''

    df = pd.read_csv(dataset)
    train_df, val_df = train_test_split(df, test_size=val_size, shuffle=True)

    return train_df, val_df


def concatPILImage(patches):
    # this function only supports 3x3 now.
    assert len(patches) == 9
    h, w = patches[0].size
    target = Image.new('RGB', (h * 3, w * 3))
    for i in range(len(patches)):
        a = h * (i % 3)
        b = w * (i // 3)
        c = h * (i % 3) + w
        d = w * (i // 3) + h
        target.paste(patches[i], (a, b, c, d))
    return target


class JigsawDataset(Dataset):
    def __init__(self, df, jig_classes=100, img_transformer=None, tile_transformer=None,
                 bias_whole_image=None, normalize=None, args=None):
        self.args = args
        self.df = df

        self.N = len(self.df)
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        self.normalize = normalize

        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        self.toTensor = transforms.ToTensor()

    def get_tile(self, img, n):

        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        return tile

    def get_image(self, index):
        
        row = self.df.iloc[index]
        smiles = row['smiles']

        try:
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
            img = img.convert('RGB')
        except Exception as e:
            print(f"Cannot convert SMILES to image: {smiles}. Error: {e}. Creating a dummy image.")
            img = Image.new('RGB', (224, 224))

        if self._image_transformer is not None:
            img = self._image_transformer(img)
        else:
            img = self.toTensor(img)

        return img

    def get_tile_data(self, img, index):
        img = img.resize((222, 222))

        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids

        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = concatPILImage(data).resize((224, 224))
        data = self._augment_tile(data)

        return data, int(order)

    def get_mask_data(self, data_non_mask, data64_non_mask,
                      mask_type, mask_shape_h, mask_shape_w, mask_ratio):
        c, h, w = data_non_mask.shape
        if mask_type == "random_mask":
            mask_matrix = mask_utils.create_random_mask(shape=(1, h, w), mask_ratio=mask_ratio)[0]
            mask64_matrix = mask_utils.create_random_mask(shape=(1, data64_non_mask.shape[1], data64_non_mask.shape[2]),
                                                          mask_ratio=mask_ratio)[0]
        elif mask_type == "rectangle_mask":
            mask_matrix = mask_utils.create_rectangle_mask(shape=(1, h, w),
                                                           mask_shape=(mask_shape_h, mask_shape_w))[
                0]
            mask64_matrix = \
                mask_utils.create_rectangle_mask(shape=(1, data64_non_mask.shape[1], data64_non_mask.shape[2]),
                                                 mask_shape=(mask_shape_h, mask_shape_w))[0]
        elif mask_type == "mix_mask":
            if random() > 0.5:
                mask_matrix = \
                    mask_utils.create_random_mask(shape=(1, data64_non_mask.shape[1], data64_non_mask.shape[2]),
                                                  mask_ratio=mask_ratio)[0]
            else:
                mask_matrix = \
                    mask_utils.create_rectangle_mask(shape=(1, data64_non_mask.shape[1], data64_non_mask.shape[2]),
                                                     mask_shape=(mask_shape_h, mask_shape_w))[0]
        # starting mask
        data_mask = data_non_mask.clone()
        data64_mask = data64_non_mask.clone()
        for i in range(3):  # 3 channels
            data_mask[i][torch.from_numpy(mask_matrix) == 1] = torch.mean(data_mask[i])
            data64_mask[i][torch.from_numpy(mask64_matrix) == 1] = torch.mean(data64_mask[i])

        return data_mask, data64_mask

    def __getitem__(self, index):
        img = self.get_image(index)
        img64 = img.resize((64, 64))

        data, order = self.get_tile_data(img, index)

        # get mask data
        data_non_mask = self._augment_tile(img)
        data64_non_mask = self._augment_tile(img64)
        cl_data_mask, cl_data64_mask = self.get_mask_data(data_non_mask, data64_non_mask,
                                                          mask_type=self.args.cl_mask_type,
                                                          mask_shape_h=self.args.cl_mask_shape_h,
                                                          mask_shape_w=self.args.cl_mask_shape_w,
                                                          mask_ratio=self.args.cl_mask_ratio)

        if self.normalize != None:
            data, data64_non_mask = self.normalize(data), self.normalize(data64_non_mask)
            data_non_mask = self.normalize(data_non_mask)
            cl_data_mask, cl_data64_mask = self.normalize(cl_data_mask), self.normalize(cl_data64_mask)

        return data, order, data_non_mask, data64_non_mask, cl_data_mask, cl_data64_mask

    def __len__(self):
        return self.N

    def __retrieve_permutations(self, classes):

        all_perm = np.load('permutations_%d.npy' % (classes))

        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

