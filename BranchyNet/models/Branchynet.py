import torch
import torch.nn as nn

#import numpy as np
#from scipy.stats import entropy


class ConvPoolAc(nn.Module):
    def __init__(self, chanIn, chanOut, kernel=3, stride=1, padding=1, p_ceil_mode=False):
        super(ConvPoolAc, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=kernel,
                stride=stride, padding=padding, bias=False),
            nn.MaxPool2d(2, stride=2, ceil_mode=p_ceil_mode),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)

class Branchynet(nn.Module):
    def __init__(self, exit_threshold=0.5):
        super(Branchynet, self).__init__()

        self.fast_inference_mode = False
        self.exit_threshold = torch.tensor([exit_threshold], dtype=torch.float32)

        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.exit_loss_weights = [1.0, 0.3]

        self.chansIn = [5,10]
        self.chansOut = [10,20]

        self._build_backbone()
        self._build_exits()

    def _build_backbone(self):
        strt_bl = ConvPoolAc(1, 5, kernel=5, stride=1, padding=4)
        self.backbone.append(strt_bl)

        bb_layers = []
        bb_layers.append(ConvPoolAc(5, 10,
                            kernel=5, stride=1, padding=4) )
        bb_layers.append(ConvPoolAc(10, 20,
                            kernel=5, stride=1, padding=3) )
        bb_layers.append(nn.Flatten())
        bb_layers.append(nn.Linear(720, 84))

        remaining_backbone_layers = nn.Sequential(*bb_layers)
        self.backbone.append(remaining_backbone_layers)

    def _build_exits(self):
        ee1 = nn.Sequential(
            ConvPoolAc(5, 10, kernel=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(640,10)
        )
        self.exits.append(ee1)

        eeF = nn.Sequential(
            nn.Linear(84,10),
        )

        self.exits.append(eeF)

    def exit_criterion(self, x):
        with torch.no_grad():
            entr = -torch.sum(pk * torch.log(x))
            return entr < self.exit_threshold

    def exit_criterion_top1(self, x):
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            top1 = torch.max(pk)
            return top1 > self.exit_threshold

    @torch.jit.unused #decorator to skip jit comp
    def _forward_training(self, x):
        res = []
        for bb, ee in zip(self.backbone, self.exits):
            x = bb(x)
            res.append(ee(x))
        return res

    def forward(self, x):
        if self.fast_inference_mode:
            for bb, ee in zip(self.backbone, self.exits):
                x = bb(x)
                res = ee(x)
                if self.exit_criterion_top1(res):
                    return res
            return res
        else:
            return self._forward_training(x)

    def set_fast_inf_mode(self, mode=True):
        if mode:
            self.eval()
        self.fast_inference_mode = mode