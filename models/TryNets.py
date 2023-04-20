import torch.nn as nn

'''
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
TryNet01 --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       50,176
│    └─ReLU: 2-2                         --
├─Sequential: 1-2                        --
│    └─Conv2d: 2-3                       960
│    └─BatchNorm2d: 2-4                  192
│    └─ReLU: 2-5                         --
│    └─MaxPool2d: 2-6                    --
│    └─Conv2d: 2-7                       332,160
│    └─BatchNorm2d: 2-8                  768
│    └─ReLU: 2-9                         --
│    └─MaxPool2d: 2-10                   --
│    └─Conv2d: 2-11                      1,228,928
│    └─BatchNorm2d: 2-12                 256
│    └─ReLU: 2-13                        --
│    └─MaxPool2d: 2-14                   --
├─Sequential: 1-3                        --
│    └─Flatten: 2-15                     --
│    └─Linear: 2-16                      2,306
=================================================================
Total params: 1,615,746
Trainable params: 1,615,746
Non-trainable params: 0
=================================================================

Epoch:  4 Batch: 3121 Loss: 0.3195 Accuracy Train: 84.152%

Duration: 2844 seconds
'''
class TryNet01(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.expand_to = 32

        self.adaptation = nn.Sequential(
            nn.Linear(48, self.expand_to ** 2),
            nn.ReLU(),)
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            
            nn.Conv2d(96, 384, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            
            nn.Conv2d(384, 128, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2),           
        )
        
        self.exit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, num_classes),)
    
    def unflat(self, x):
        return x.view(-1, 1, self.expand_to, self.expand_to)
    
    def forward(self, x):
        x = self.unflat(self.adaptation(x))
        return self.exit(self.backbone(x))

'''
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
TryNet02                                 --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       50,176
│    └─ReLU: 2-2                         --
├─Sequential: 1-2                        --
│    └─Conv2d: 2-3                       960
│    └─BatchNorm2d: 2-4                  192
│    └─ReLU: 2-5                         --
│    └─MaxPool2d: 2-6                    --
│    └─Conv2d: 2-7                       614,656
│    └─BatchNorm2d: 2-8                  512
│    └─ReLU: 2-9                         --
│    └─MaxPool2d: 2-10                   --
├─Sequential: 1-3                        --
│    └─Flatten: 2-11                     --
│    └─Linear: 2-12                      8,194
=================================================================
Total params: 674,690
Trainable params: 674,690
Non-trainable params: 0
=================================================================

Epoch:  4 Batch: 3121 Loss: 0.4033 Accuracy Train: 84.835%

Duration: 954 seconds
'''
class TryNet02(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.expand_to = 32

        self.adaptation = nn.Sequential(
            nn.Linear(48, self.expand_to ** 2),
            nn.ReLU(),)
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
                       
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2),           
        )
        
        self.exit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, num_classes),)
    
    def unflat(self, x):
        return x.view(-1, 1, self.expand_to, self.expand_to)
    
    def forward(self, x):
        x = self.unflat(self.adaptation(x))
        return self.exit(self.backbone(x))




class TryNet03(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.expand_to = 64

        self.adaptation = nn.Sequential(
            nn.Linear(48, self.expand_to ** 2),
            nn.ReLU(),)
        
        self.backbone = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=1)),
                       
            nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=1)),   
            
            nn.Sequential(
                nn.Conv2d(256, 96, kernel_size=7, stride=2, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=7, stride=2)),            
        )
        
        self.exit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9600, num_classes),
        )
    
    def unflat(self, x):
        return x.view(-1, 1, self.expand_to, self.expand_to)
    
    def forward(self, x):
        x = self.unflat(self.adaptation(x))
        return self.exit(self.backbone(x))        