import torch
import torch.nn as nn
import time

class MawiNetLayers():
    def __init__(self, num_classes=2):
        self.backbone = [
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU()),
            
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU()),
            
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU()),
        ]

        self.exits = [
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(800, num_classes)),
            
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024, num_classes)),

            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1152, num_classes))
        ]

class MawiNetee0(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        layers = MawiNetLayers(num_classes=num_classes)

        self.backbone = layers.backbone[0]
        self.exit     = layers.exits[0]

    def forward(self, x):
        return self.exit(self.backbone(x))

class MawiNetee1(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        layers = MawiNetLayers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone[0:2])
        self.exit     = layers.exits[1]

    def forward(self, x):
        return self.exit(self.backbone(x))

class MawiNetFlat(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(48, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        return self.model(x)

class MawiNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        layers = MawiNetLayers(num_classes=num_classes)

        self.backbone = nn.Sequential(*layers.backbone[0:3])
        self.exit     = layers.exits[2]
        
    def forward(self, x):
        return self.exit(self.backbone(x))

class MawiNetWithExits(nn.Module):
    def __init__(self, num_classes=2, exit_loss_weights=[1.0, 0.5, 0.2]):
        super().__init__()

        layers = MawiNetLayers(num_classes=num_classes)

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