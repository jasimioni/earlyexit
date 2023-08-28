# https://blog.paperspace.com/alexnet-pytorch/

import torch
import torch.nn as nn
import time

# Adjusted for MNIST
class AlexNetWithExitsMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetWithExitsMNIST, self).__init__()
        self.backbone = nn.ModuleList() 
        self.exits = nn.ModuleList()
        self.exit_threshold = torch.tensor([0.5, 0.7], dtype=torch.float32)
        self.fast_inference_mode = False
        self.exit_loss_weights = [1.0, 0.5, 0.2]

        self.backbone.append(
            nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 1))
        )

        self.exits.append(nn.Sequential(
            nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2)),
            
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2)),
            
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(4096, num_classes))
        ))

        self.backbone.append(nn.Sequential(
            nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2)),

            nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU())
        ))

        self.exits.append(nn.Sequential(
            nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2)),
            
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(4096, num_classes))
        ))

        self.backbone.append(nn.Sequential(
            nn.Sequential(
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU()),

            nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2))
        ))

        self.exits.append(nn.Sequential(
            nn.Flatten(),

            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 2048),
                nn.ReLU()),
            
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 2048),
                nn.ReLU()),
            
            nn.Sequential(
                nn.Linear(2048, num_classes))
        ))

    def exit_criterion(self, ee_n, x):
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            nc = torch.max(pk)
            return nc > self.exit_threshold[ee_n]

    @torch.jit.unused #decorator to skip jit comp
    def _forward_training(self, x):
        res = []
        for bb, ee in zip(self.backbone, self.exits):
            x = bb(x)
            res.append(ee(x))
        return res

    def forward(self, x):
        if self.fast_inference_mode:
            for ee_n, (bb, ee) in enumerate(zip(self.backbone, self.exits)):
                x = bb(x)
                res = ee(x)
                if self.exit_criterion(ee_n, x):
                    return [res, 'ee' + ee_n]
            return [res, 'main']
        else:
            return self._forward_training(x)

    def set_fast_inference_mode(self, mode=True):
        if mode:
            self.eval()
        self.fast_inference_mode = mode        

class AlexNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=0), # Changed from 3 to 1 - FashionMNIST
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(2048, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class AlexNetMNISTee2(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetMNISTee2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=0), # Changed from 3 to 1 - FashionMNIST
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class AlexNetMNISTee1(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetMNISTee1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=0), # Changed from 3 to 1 - FashionMNIST
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc= nn.Sequential(
            nn.Linear(4096, num_classes))
       
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2304, 1024),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(1024, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class AlexNetCIFAR10Layers():
    def __init__(self, num_classes=10):
        self.backbone = [
            nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2)),
            
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3, stride = 2)),
                nn.Sequential(
                    nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(384),
                    nn.ReLU())),
            
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(384),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3, stride = 2)))
        ]

        self.exits = [
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3, stride = 2)),
                
                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3, stride = 2)),
                
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1024, num_classes))),
            
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3, stride = 2)),
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2304, num_classes))),

            nn.Sequential(
                nn.Flatten(),
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(2304, 1024),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(1024, 1024),
                    nn.ReLU()),
                nn.Linear(1024, num_classes))
        ]

class AlexNetCIFAR10ee1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        layers = AlexNetCIFAR10Layers(num_classes=num_classes)

        self.backbone = layers.backbone[0]
        self.exit     = layers.exits[0]

    def forward(self, x):
        return self.exit(self.backbone(x))

class AlexNetCIFAR10ee2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        layers = AlexNetCIFAR10Layers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone[0:2])
        self.exit     = layers.exits[1]

    def forward(self, x):
        return self.exit(self.backbone(x))

class AlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        layers = AlexNetCIFAR10Layers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone[0:3])
        self.exit     = layers.exits[2]
        
    def forward(self, x):
        return self.exit(self.backbone(x))

class AlexNetWithExitsCIFAR10(nn.Module):
    def __init__(self, num_classes=10, exit_loss_weights=[1.0, 0.5, 0.2]):
        super().__init__()

        layers = AlexNetCIFAR10Layers(num_classes=num_classes)

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
      

class AlexNetMawiLayers():
    def __init__(self, num_classes=2):

        self.adaptation = nn.Sequential(
                nn.Linear(48, 4096),
                nn.ReLU(),
        )

        self.backbone = [
            nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=7, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2)),
            
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 5, stride = 2)),
                nn.Sequential(
                    nn.Conv2d(256, 384, kernel_size=5, stride=1, padding=1),
                    nn.BatchNorm2d(384),
                    nn.ReLU())),
            
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(384),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3, stride = 2)))
        ]

        self.exits = [
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 5, stride = 2)),
                
                nn.Sequential(nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 5, stride = 2)),
                
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1024, num_classes))),
            
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=5, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 5, stride = 2)),
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1024, num_classes))),

            nn.Sequential(
                nn.Flatten(),
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(4096, 1024),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(1024, 1024),
                    nn.ReLU()),
                nn.Linear(1024, num_classes))
        ]

class AlexNetMawiee1(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        layers = AlexNetMawiLayers(num_classes=num_classes)

        self.adaptation = layers.adaptation
        self.backbone = layers.backbone[0]
        self.exit     = layers.exits[0]

    def forward(self, x):
        x = self.adaptation(x).view(-1, 1, 64, 64)
        return self.exit(self.backbone(x))

class AlexNetMawiee2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        layers = AlexNetMawiLayers(num_classes=num_classes)

        self.adaptation = layers.adaptation
        self.backbone = nn.Sequential(*layers.backbone[0:2])
        self.exit     = layers.exits[1]

    def forward(self, x):
        x = self.adaptation(x).view(-1, 1, 64, 64)
        return self.exit(self.backbone(x))

class AlexNetMawi(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        layers = AlexNetMawiLayers(num_classes=num_classes)

        self.adaptation = layers.adaptation
        self.backbone = nn.Sequential(*layers.backbone[0:3])
        self.exit     = layers.exits[2]
        
    def forward(self, x):
        x = self.adaptation(x).view(-1, 1, 64, 64)
        return self.exit(self.backbone(x))

class AlexNetWithExitsMawi(nn.Module):
    def __init__(self, num_classes=2, exit_loss_weights=[1.0, 0.5, 0.2]):
        super().__init__()

        layers = AlexNetMawiLayers(num_classes=num_classes)

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
