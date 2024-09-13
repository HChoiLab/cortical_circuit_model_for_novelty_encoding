import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import torch
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
from torch.nn.functional import avg_pool2d, max_pool2d

INTERPOLATION_MODES = {
    'nearest': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC
}

def extract_features(data: np.ndarray,
                     method: str = 'pca',
                     num_pcs: int = 3,
                     vectorize: bool = True,
                     new_size: tuple[int] = None,
                     downsample_mode: str = 'bilinear'):
    """
    Embeds stimuli into a lower-dimensional feature space
    :param data: array of shape (HxW), (CxHxW), (NxHxW), or (NxCxHxW) where N = num of images, C = num of channels,
            H = image height, and W = image width. If the array has > 2 dimensions, 1st dim is assumed to be a batch dim
    :param method: feature extraction method, options are:
            - 'pca': uses PCA to linearly project images into a lower-dimensional subspace
            - 'downsample': simply down-samples the images. if this is specified, then four other keyword argument must
                            be given which are described below
    :param num_pcs: in case method = 'pca', indicates the dimensionality of the new subspace
    :param vectorize: in case method = 'downsample', whether to return a vectorized version of the downsampled image
    :param new_size: in case method = 'downsample', size the downsampled image
    :param downsample_mode: in case method = 'downsample', which mode to use for downsampling, options are:
            - 'bilinear': bilinear interpolation
            - 'nearest': nearest neighbour interpolation
            - 'bicubic': bicubic interpolation
            - 'avg_pooling': 2D average pooling
            - 'max_pooling': 2D max pooling

    :return: (ndarray) new array with the features of the original images
    """

    if len(data.shape) == 2:  # if un-batched, add an extra dummy dimension for consistency and a channel dimension
        data = data.reshape((1, 1) + data.shape)
    elif len(data.shape) == 3:  # batched but no channels, add a channel dimension
        data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2]))

    N, C, H, W = data.shape

    if method == 'pca':
        # vectorize the data first
        data_flat = data.reshape((N, -1))
        # perform pca
        pca_model = PCA(n_components=num_pcs)
        features = pca_model.fit_transform(data_flat)

        return features

    elif method == 'downsample':
        # convert to tensor
        data = torch.Tensor(data)

        if downsample_mode in INTERPOLATION_MODES.keys():
            # resize transform with pytorch
            resizer = Resize(size=new_size, interpolation=INTERPOLATION_MODES[downsample_mode])
            features = resizer(data)
        else:
            # define the kernel size for pooling
            kernel_sz = (H // new_size[0], W // new_size[1])

            # define the pooling function
            pooling_fn = avg_pool2d if downsample_mode == 'avg_pooling' else max_pool2d

            # perform pooling
            features = pooling_fn(data, kernel_sz)

        # vectorize
        if vectorize:
            features = features.reshape(N, -1)

        return features.detach().cpu().numpy()
