from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from concept.model import ModelWrapper


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.scaler1 = ScaleLayer(init_value=torch.tensor(2.))
        self.scaler2 = ScaleLayer(init_value=torch.tensor(2.))
        self.scaler3 = ScaleLayer(init_value=torch.tensor(2.))

    def forward(self, x):
        x = self.scaler1(x)
        x = self.scaler2(x)
        x = self.scaler3(x)
        return x


class TestConvNet(nn.Module):
    def __init__(self):
        super(TestConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelTest(ModelWrapper):
    def __init__(self, model=None, state_dict_path=None):
        super(ModelTest, self).__init__(model, state_dict_path)


class test_model(unittest.TestCase):
    def setUp(self):
        self.net = TestNet()
        self.conv_net = TestConvNet()
        self.ckpt_dir = '/tmp/ckpts/'
        self.tmp_dirs = [self.ckpt_dir]
        for d in self.tmp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
        torch.save(self.net.state_dict(), self.ckpt_dir + 'model.pth')
        torch.save(self.conv_net.state_dict(), self.ckpt_dir + 'conv_model.pth')

        self.bottlenecks = ['scaler']

    def tearDown(self):
        for d in self.tmp_dirs:
            shutil.rmtree(d)

    def _check_output(self, state_dict_path):
        model = ModelTest(model=self.net,
                          state_dict_path=state_dict_path)
        out = model.get_predictions(torch.tensor([2.]))
        self.assertEqual(out, [16.])

    def test_try_loading_model_from_path(self):
        self._check_output(self.ckpt_dir + 'model.pth')

    def test_setting_fw_hook_bottleneck_tensors(self):
        bottlnecks = ['scaler1', 'scaler2', 'scaler3']
        model = ModelTest(self.net)
        fw_buffer = model._setting_fw_hook_bottleneck(bottlnecks)
        _ = model.model(torch.tensor([2.]))
        self.assertEqual(fw_buffer['scaler1'][0], 4.)
        self.assertEqual(fw_buffer['scaler2'][0], 8.)
        self.assertEqual(fw_buffer['scaler3'][0], 16.)

    def test_setting_bw_hook_bottleneck_tensors(self):
        bottlnecks = ['scaler1', 'scaler2', 'scaler3']
        model = ModelTest(self.net)
        bw_buffer = model._setting_bw_hook_bottleneck(bottlnecks)
        out = model.model(torch.tensor([2.]))
        out.backward()
        self.assertEqual(bw_buffer['scaler1'][0], 4.)
        self.assertEqual(bw_buffer['scaler2'][0], 2.)
        self.assertEqual(bw_buffer['scaler3'][0], 1.)

    def test_fw_bw_shape(self):
        bottlenecks = ['conv1', 'conv2', 'error']
        model = ModelTest(self.conv_net)
        fw_buffer = model._setting_fw_hook_bottleneck(bottlenecks)
        bw_buffer = model._setting_bw_hook_bottleneck(bottlenecks)
        loss = model.model(torch.rand((2, 3, 32, 32)))[0, 0]
        loss.backward()
        self.assertEqual(fw_buffer['conv1'][0].shape, bw_buffer['conv1'][0].shape)
        self.assertEqual(fw_buffer['conv2'][0].shape, bw_buffer['conv2'][0].shape)

    def test_get_gradient(self):
        bottlenecks = ['conv1', 'conv2', 'error']
        model = ModelTest(self.conv_net)
        examples = [torch.rand((3, 3, 32, 32)), torch.rand((2, 3, 32, 32))]
        gradients = model.get_gradient(examples=examples, y=0, bottlenecks=bottlenecks)
        self.assertEqual(len(gradients['conv1']), 5)

    def test_get_acts(self):
        bottlenecks = ['scaler1', 'scaler2']
        examples = [
            torch.tensor([1., 1.]),
            torch.tensor([2.])
        ]
        model = ModelTest(self.net)
        acts = model.get_acts(examples=examples, bottlenecks=bottlenecks)
        self.assertEqual(acts['scaler1'][0], 2.)
        self.assertEqual(acts['scaler1'][2], 4.)

if __name__ == "__main__":
    unittest.main()
