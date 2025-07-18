import numpy as np
import random
import os
from PIL import Image
import torch

def make_training_sequence(im1: np.ndarray, im2: np.ndarray,
                           blank_ts: int = 1, im_ts: int = 2,
                           num_before: int = 3,
                           num_after: int = 3,
                           omission_index: int = -1):
    # construct the sequence array
    seq_length = (im_ts + blank_ts) * (num_before + num_after)
    sequence = np.zeros((len(im1), seq_length))

    before_onsets = []
    before_offsets = []
    after_onsets = []
    after_offsets = []

    # populate the sequence array
    for i in range(num_before):
        if omission_index == i:
            continue
        sequence[:, i * (blank_ts + im_ts) + blank_ts: (i + 1) * (blank_ts + im_ts)] = im1.repeat(im_ts, axis=-1)
        before_onsets.append(i * (blank_ts + im_ts) + blank_ts)
        before_offsets.append((i + 1) * (blank_ts + im_ts))

    for j in range(num_before, num_before + num_after):
        if omission_index == j:
            continue
        sequence[:, j * (blank_ts + im_ts) + blank_ts: (j + 1) * (blank_ts + im_ts)] = im2.repeat(im_ts, axis=-1)
        after_onsets.append(j * (blank_ts + im_ts) + blank_ts)
        after_offsets.append((j + 1) * (blank_ts + im_ts))

    return sequence.T, {'before': (before_onsets, before_offsets), 'after': (after_onsets, after_offsets)}


def get_sequences(image_list, num_presentations=4, blank_ts=4, pres_ts=6, min_before_change=2, omission_prob=0.3, seed=None):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    im_ts, seqs, omission_ts = [], [], []
    
    for i1 in range(len(image_list)):
        for i2 in range(len(image_list)):
            if i1 != i2:
                
                im1, im2 = image_list[i1], image_list[i2]
                
                for nbf in range(min_before_change, num_presentations - 1):
                    
                    # decide if this sequence will contain an omission
                    omit_indx = -1
                    u = np.random.rand(1)
                    if u < omission_prob:
                        omit_indx = np.random.choice(np.setdiff1d(np.arange(min_before_change, num_presentations), nbf), 1).item()
                    
                    sq, ts = make_training_sequence(im1.reshape((-1, 1)),
                                                    im2.reshape((-1, 1)),
                                                    blank_ts=blank_ts, im_ts=pres_ts,
                                                    num_before=nbf,
                                                    num_after=num_presentations - nbf,
                                                    omission_index=omit_indx)
                    
                    im_ts.append(ts)
                    seqs.append(sq)
                    omission_ts.append(omit_indx * (blank_ts + pres_ts) + blank_ts)
                    
    seqs = np.stack(seqs, axis=0)

    return seqs, im_ts, omission_ts


def get_sequences_depr(image_list, num_presentations=4, blank_ts=4, pres_ts=6, min_before_change=2, omission_prob=0.3, seed=None):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    init_ims = image_list * (len(image_list) - 1)

    change_ims = image_list[1:]
    for i in range(1, len(image_list)):
        change_ims = change_ims + image_list[:i] + image_list[i + 1:]

    num_befores = np.random.randint(min_before_change, num_presentations - 1, size=len(init_ims))

    im_ts, seqs, omission_ts = [], [], []

    for im1, im2, nbf in zip(init_ims, change_ims, num_befores):
        
        # decide if this sequence will contain an omission
        omit_indx = -1
        u = np.random.rand(1)
        if u < omission_prob:
            omit_indx = np.random.choice(np.setdiff1d(np.arange(min_before_change, num_presentations), nbf), 1).item()            
        
        sq, ts = make_training_sequence(im1.reshape((-1, 1)),
                                        im2.reshape((-1, 1)),
                                        blank_ts=blank_ts, im_ts=pres_ts,
                                        num_before=nbf,
                                        num_after=num_presentations - nbf,
                                        omission_index=omit_indx)

        im_ts.append(ts)
        seqs.append(sq)
        omission_ts.append(omit_indx * (blank_ts + pres_ts) + blank_ts)

    seqs = np.stack(seqs, axis=0)

    return seqs, im_ts, omission_ts


def load_images_to_numpy_array(path, size=None, grayscale=True, num_images=20):
    """
    Load all images from the specified directory into a NumPy array.

    Parameters:
    path (str): The directory containing images.
    size (tuple): Optional. Resize image to the given size (width, height).

    Returns:
    numpy.ndarray: An array of images of shape (H, W, C).
    """
    files = os.listdir(path)
    images = []
    for i in range(num_images):
        file = files[i]
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(path, file)
            with Image.open(img_path) as img:
                if grayscale:
                    img = img.convert('L')
                if size:
                    img = img.resize(size, Image.ANTIALIAS)
                img = np.asarray(img)
                # Transpose the image dimensions if it's not grayscale
                if img.ndim == 3:
                    img = img.transpose(2, 0, 1)
                images.append(img)

    return np.array(images)


def load_results_files(directory, prefix):
    # Ensure directory exists
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return {}

    # Get list of files in directory
    files = os.listdir(directory)
    # Filter files by prefix and ending with '.pt'
    torch_files = [file for file in files if file.startswith(prefix)]

    args = None

    change_responses = {"familiar": {}, "novel": {}, "familiar_means":{}, "novel_means": {}}
    omission_responses = {"familiar": {}, "novel": {}, "familiar_means": {}, "novel_means": {}}
    training_progress = {}
    for i, file in enumerate(torch_files):
        file_path = os.path.join(directory, file)
        try:
            # Load Torch file
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            # Concatenate tensors along the first axis

            if args is None:
                args = data['args']

            # first for change responses
            for key in data["change_responses"]["familiar"].keys():
                fam_value = data["change_responses"]["familiar"][key]
                nov_value = data["change_responses"]["novel"][key]
                fam_mean = data["change_responses"]["familiar_means"][key]
                nov_mean = data["change_responses"]["novel_means"][key]
                if key not in change_responses["familiar"]:
                    change_responses["familiar"][key] = fam_value
                    change_responses["novel"][key] = nov_value
                    change_responses["familiar_means"][key] = fam_mean
                    change_responses["novel_means"][key] = nov_mean
                else:
                    change_responses["familiar"][key] = torch.cat([change_responses["familiar"][key], fam_value])
                    change_responses["novel"][key] = torch.cat([change_responses["novel"][key], nov_value])
                    change_responses["familiar_means"][key] = torch.cat([change_responses["familiar_means"][key], fam_mean])
                    change_responses["novel_means"][key] = torch.cat([change_responses["novel_means"][key], nov_mean])
            
            # now for omission responses
            for key in data["omission_responses"]["familiar"].keys():
                fam_value = data["omission_responses"]["familiar"][key]
                nov_value = data["omission_responses"]["novel"][key]

                if "familiar_means" in data['omission_responses'].keys():
                    fam_mean = data["omission_responses"]["familiar_means"][key]
                    nov_mean = data["omission_responses"]["novel_means"][key]
                if key not in omission_responses["familiar"]:
                    omission_responses["familiar"][key] = fam_value
                    omission_responses["novel"][key] = nov_value

                    if "familiar_means" in data['omission_responses'].keys():
                        omission_responses["familiar_means"][key] = fam_mean
                        omission_responses["novel_means"][key] = nov_mean
                else:
                    omission_responses["familiar"][key] = torch.cat([omission_responses["familiar"][key], fam_value])
                    omission_responses["novel"][key] = torch.cat([omission_responses["novel"][key], nov_value])

                    if "familiar_means" in data['omission_responses'].keys():
                        omission_responses["familiar_means"][key] = torch.cat([omission_responses["familiar_means"][key], fam_mean])
                        omission_responses["novel_means"][key] = torch.cat([omission_responses["novel_means"][key], nov_mean])
            
            # finally training progress
            for key in data['training_progress'].keys():
                if key not in training_progress:
                    training_progress[key] = [data['training_progress'][key]]
                else:
                    training_progress[key] += [data['training_progress'][key]]

            print(f"Loaded {i + 1}/{len(torch_files)} files", end='\r')
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    training_progress = {k: np.stack(v) for k, v in training_progress.items()}
    
    return args, change_responses, omission_responses, training_progress