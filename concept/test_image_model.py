from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from concept.model import ImageModelWrapper


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class test_image_model(unittest.TestCase):
    def setUp(self) -> None:
        self.net = Net()
        self.transform = transforms.Compose([
            transforms.Resize(28, 28),
            transforms.ToTensor()
        ])
        self.model = ImageModelWrapper(model=self.net,
                                       state_dict_path=None,
                                       image_shape=(28, 28),
                                       labels_path='/home/dizzy/workspace/auto-concept-extractor/data/01_raw/AwA2/_classes.txt')

    def test_id_to_label(self):
        self.assertEqual(self.model.id_to_label(0), 'antelope')

    def test_label_to_id(self):
        self.assertEqual(self.model.label_to_id('antelope'), 0)

if __name__ == "__main__":
    unittest.main()
