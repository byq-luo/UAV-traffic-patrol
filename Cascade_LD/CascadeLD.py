import sys

path = __file__.split('CascadeLD.py')[0]
sys.path.append(path)

import torch
import torchvision
import numpy as np

# Data loading and visualization imports
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt

# Model loading
from models.erfnet import Net as ERFNet
from models.lcnet import Net as LCNet

# to cuda or not to cuda
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'

# Descriptor size definition
DESCRIPTOR_SIZE = 64

# Maximum number of lanes the network has been trained with + background
NUM_CLASSES_SEGMENTATION = 5

# Maximum number of classes for classification
NUM_CLASSES_CLASSIFICATION = 3


def extract_descriptors(label, image):
    # avoids problems in the sampling
    eps = 0.01

    # The labels indices are not contiguous e.g. we can have index 1, 2, and 4 in an image
    # For this reason, we should construct the descriptor array sequentially
    inputs = torch.zeros(0, 3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    # This is needed to keep track of the lane we are classifying
    mapper = {}
    classifier_index = 0

    # Iterating over all the possible lanes ids
    for i in range(1, NUM_CLASSES_SEGMENTATION):
        # This extracts all the points belonging to a lane with id = i
        single_lane = label.eq(i).view(-1).nonzero(as_tuple=False).squeeze()

        # As they could be not continuous, skip the ones that have no points
        if single_lane.numel() == 0 or len(single_lane.size()) == 0:
            continue

        # Points to sample to fill a squared desciptor
        sample = torch.zeros(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE)
        if torch.cuda.is_available():
            sample = sample.cuda()

        sample = sample.uniform_(0, single_lane.size()[0] - eps).long()
        sample, _ = sample.sort()

        # These are the points of the lane to select
        points = torch.index_select(single_lane, 0, sample)

        # First, we view the image as a set of ordered points
        descriptor = image.squeeze().view(3, -1)

        # Then we select only the previously extracted values
        descriptor = torch.index_select(descriptor, 1, points)

        # Reshape to get the actual image
        descriptor = descriptor.view(3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
        descriptor = descriptor.unsqueeze(0)

        # Concatenate to get a batch that can be processed from the other network
        inputs = torch.cat((inputs, descriptor), 0)

        # Track the indices
        mapper[classifier_index] = i
        classifier_index += 1

    return inputs, mapper


def lane_detect(im_tensor):
    # Image size
    _, HEIGHT, WIDTH = im_tensor.shape
    im_tensor = im_tensor.unsqueeze(0)

    # Creating CNNs and loading pretrained models
    segmentation_network = ERFNet(NUM_CLASSES_SEGMENTATION)
    classification_network = LCNet(NUM_CLASSES_CLASSIFICATION, DESCRIPTOR_SIZE,
                                   DESCRIPTOR_SIZE)

    segmentation_network.load_state_dict(
        torch.load(path + 'pretrained/erfnet_tusimple.pth',
                   map_location=map_location))
    model_path = path + 'pretrained/classification_{}_{}class.pth'.format(
        DESCRIPTOR_SIZE, NUM_CLASSES_CLASSIFICATION)
    classification_network.load_state_dict(
        torch.load(model_path, map_location=map_location))

    segmentation_network = segmentation_network.eval()
    classification_network = classification_network.eval()

    if torch.cuda.is_available():
        segmentation_network = segmentation_network.cuda()
        classification_network = classification_network.cuda()
        im_tensor = im_tensor.cuda()

    out_segmentation = segmentation_network(im_tensor)
    out_segmentation = out_segmentation.max(dim=1)[1]

    out_segmentation_np = out_segmentation.cpu().numpy()[0]
    descriptors, index_map = extract_descriptors(out_segmentation, im_tensor)
    classes = classification_network(descriptors).max(1)[1]

    lane_map = torch.zeros(HEIGHT, WIDTH, dtype=torch.int64)
    if torch.cuda.is_available():
        lane_map = lane_map.cuda()
    for i, lane_index in index_map.items():
        lane_map[out_segmentation_np == lane_index] = classes[i] + 1

    return lane_map


if __name__ == '__main__':
    im = Image.open('./test.png')
    im = im.resize((640, 360))
    A = ToTensor()(im)
    print(A.shape)
    print(A.type())

    B = lane_detect(A)

    plt.figure("test")
    plt.imshow(B.cpu())
    plt.show()
