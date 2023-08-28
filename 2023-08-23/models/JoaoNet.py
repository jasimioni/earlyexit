# https://blog.paperspace.com/alexnet-pytorch/

import torch
import torch.nn as nn
import time

class JoaoNetCIFAR10Layers():
    def __init__(self, num_classes=10):
        self.backbone = [
            nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2)),
            
            nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2)),
            
            nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU())
        ]

        self.exits = [
            nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=2),
                nn.Flatten(),
                nn.Linear(2400, num_classes)),
            
            nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=2),
                nn.Flatten(),
                nn.Linear(1024, num_classes)),

            nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=2),
                nn.Flatten(),
                nn.Linear(1024, num_classes))
        ]

class JoaoNetCIFAR10ee0(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        layers = JoaoNetCIFAR10Layers(num_classes=num_classes)

        self.backbone = layers.backbone[0]
        self.exit     = layers.exits[0]

    def forward(self, x):
        return self.exit(self.backbone(x))

class JoaoNetCIFAR10ee1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        layers = JoaoNetCIFAR10Layers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone[0:2])
        self.exit     = layers.exits[1]

    def forward(self, x):
        return self.exit(self.backbone(x))

class JoaoNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        layers = JoaoNetCIFAR10Layers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone[0:3])
        self.exit     = layers.exits[2]
        
    def forward(self, x):
        return self.exit(self.backbone(x))

class JoaoNetWithExitsCIFAR10(nn.Module):
    def __init__(self, num_classes=10, exit_loss_weights=[1.0, 0.5, 0.2]):
        super().__init__()

        layers = JoaoNetCIFAR10Layers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone)
        self.exits    = nn.Sequential(*layers.exits)

        self.exit_threshold = torch.tensor([0.5, 0.7], dtype=torch.float32)
        self.fast_inference_mode = False
        self.measurement_mode = False
        self.exit_loss_weights = exit_loss_weights

    def exit_criterion(self, ee_n, x):
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            nc = torch.max(pk)
            return nc > self.exit_threshold[ee_n]

    def _forward_all_exits(self, x):
        results = []
        for bb, ee in zip(self.backbone, self.exits):
            if self.measurement_mode:
                st = time.process_time()
                x = bb(x)
                im = time.process_time()
                res = ee(x)
                ed = time.process_time()
                results.append([ res, im - st, ed - im ])
            else:
                x = bb(x)
                results.append(ee(x))

        return results

    def forward(self, x):
        if self.fast_inference_mode:
            for ee_n, (bb, ee) in enumerate(zip(self.backbone, self.exits)):
                x = bb(x)
                res = ee(x)
                if self.exit_criterion(ee_n, res):
                    return [res, 'ee' + ee_n]
            return [res, 'main']

        return self._forward_all_exits(x)

    def exits_certainty(self, x):
        results = []
        for ee_n, (bb, ee) in enumerate(zip(self.backbone, self.exits)):
            x = bb(x)
            res = ee(x)
            certainty, predicted = torch.max(nn.functional.softmax(res, dim=-1), 1)
            results.append([ certainty.item(), predicted.item() ])
        return results

    def set_fast_inference_mode(self, mode=True):
        if mode:
            pass
            # self.eval()
        self.fast_inference_mode = mode

    def set_measurement_mode(self, mode=True):
        if mode:
            pass 
            # self.eval()
        self.measurement_mode = mode
      

        
