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

'''
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
TryNet03                                 --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       200,704
│    └─ReLU: 2-2                         --
├─Sequential: 1-2                        --
│    └─Sequential: 2-3                   --
│    │    └─Conv2d: 3-1                  960
│    │    └─BatchNorm2d: 3-2             192
│    │    └─ReLU: 3-3                    --
│    │    └─MaxPool2d: 3-4               --
│    └─Sequential: 2-4                   --
│    │    └─Conv2d: 3-5                  221,440
│    │    └─BatchNorm2d: 3-6             512
│    │    └─ReLU: 3-7                    --
│    │    └─MaxPool2d: 3-8               --
│    └─Sequential: 2-5                   --
│    │    └─Conv2d: 3-9                  1,204,320
│    │    └─BatchNorm2d: 3-10            192
│    │    └─ReLU: 3-11                   --
│    │    └─MaxPool2d: 3-12              --
├─Sequential: 1-3                        --
│    └─Flatten: 2-6                      --
│    └─Linear: 2-7                       19,202
=================================================================
Total params: 1,647,522
Trainable params: 1,647,522
Non-trainable params: 0
=================================================================

Epoch:  4 Batch: 9381 Loss: 0.3869 Accuracy Train: 84.922%

Duration: 14495 seconds
'''

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

'''
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
TryNet04                                 --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       200,704
│    └─ReLU: 2-2                         --
├─Sequential: 1-2                        --
│    └─Sequential: 2-3                   --
│    │    └─Conv2d: 3-1                  20,992
│    │    └─BatchNorm2d: 3-2             512
│    │    └─ReLU: 3-3                    --
│    │    └─MaxPool2d: 3-4               --
├─Sequential: 1-3                        --
│    └─Flatten: 2-4                      --
│    └─Linear: 2-5                       8,194
=================================================================
Total params: 230,402
Trainable params: 230,402
Non-trainable params: 0
=================================================================

Epoch:  4 Batch: 3121 Loss: 0.4130 Accuracy Train: 84.706%

Duration: 652 seconds
'''
class TryNet04(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.expand_to = 64

        self.adaptation = nn.Sequential(
            nn.Linear(48, self.expand_to ** 2),
            nn.ReLU(),)
        
        self.backbone = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=9, stride=2, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=9, stride=5)),
        )
        
        self.exit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, num_classes),
        )
    
    def unflat(self, x):
        return x.view(-1, 1, self.expand_to, self.expand_to)
    
    def forward(self, x):
        x = self.unflat(self.adaptation(x))
        return self.exit(self.backbone(x))  



'''
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
TryNet05                                 --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       200,704
│    └─ReLU: 2-2                         --
├─Sequential: 1-2                        --
│    └─Sequential: 2-3                   --
│    │    └─Conv2d: 3-1                  20,992
│    │    └─BatchNorm2d: 3-2             512
│    │    └─ReLU: 3-3                    --
│    │    └─MaxPool2d: 3-4               --
├─Sequential: 1-3                        --
│    └─Flatten: 2-4                      --
│    └─Linear: 2-5                       8,390,656
│    └─ReLU: 2-6                         --
│    └─Linear: 2-7                       4,098
=================================================================
Total params: 8,616,962
Trainable params: 8,616,962
Non-trainable params: 0
=================================================================

Epoch:  4 Batch: 3121 Loss: 0.5248 Accuracy Train: 66.770%

Duration: 750 seconds


Epoch:  0 Batch: 3111 Loss: 0.4286 Accuracy Train: 76.554%
Epoch:  0 Batch: 3121 Loss: 0.3706 Accuracy Train: 76.576%
Epoch:  1 Batch:   1 Loss: 0.3566 Accuracy Train: 86.333%
Epoch:  1 Batch:  11 Loss: 0.3851 Accuracy Train: 83.636%
...
Epoch:  1 Batch: 3111 Loss: 0.6919 Accuracy Train: 66.236%
Epoch:  1 Batch: 3121 Loss: 0.6930 Accuracy Train: 66.185%
Epoch:  2 Batch:   1 Loss: 0.6930 Accuracy Train: 51.333%
Epoch:  2 Batch:  11 Loss: 0.6911 Accuracy Train: 50.848%
...
Epoch:  3 Batch: 3111 Loss: 0.6938 Accuracy Train: 50.008%
Epoch:  3 Batch: 3121 Loss: 0.6933 Accuracy Train: 50.005%
Epoch:  4 Batch:   1 Loss: 0.6938 Accuracy Train: 44.000%
Epoch:  4 Batch:  11 Loss: 0.6937 Accuracy Train: 48.939%
'''
class TryNet05(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.expand_to = 64

        self.adaptation = nn.Sequential(
            nn.Linear(48, self.expand_to ** 2),
            nn.ReLU(),)
        
        self.backbone = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=9, stride=2, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=9, stride=5)),
        )
        
        self.exit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes),
        )
    
    def unflat(self, x):
        return x.view(-1, 1, self.expand_to, self.expand_to)
    
    def forward(self, x):
        x = self.unflat(self.adaptation(x))
        return self.exit(self.backbone(x))           

'''
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
TryNet06                                 --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       200,704
│    └─ReLU: 2-2                         --
│    └─Linear: 2-3                       16,781,312
│    └─ReLU: 2-4                         --
├─Sequential: 1-2                        --
│    └─Sequential: 2-5                   --
│    │    └─Conv2d: 3-1                  20,992
│    │    └─BatchNorm2d: 3-2             512
│    │    └─ReLU: 3-3                    --
│    │    └─MaxPool2d: 3-4               --
├─Sequential: 1-3                        --
│    └─Flatten: 2-6                      --
│    └─Linear: 2-7                       8,194
=================================================================
Total params: 17,011,714
Trainable params: 17,011,714
Non-trainable params: 0
=================================================================

Epoch:  4 Batch: 3121 Loss: 0.3590 Accuracy Train: 85.062%

Duration: 848 seconds
'''

class TryNet06(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.expand_to = 64

        self.adaptation = nn.Sequential(
            nn.Linear(48, self.expand_to ** 2),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU())
        
        self.backbone = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=9, stride=2, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=9, stride=5)),
        )
        
        self.exit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, num_classes),
        )
    
    def unflat(self, x):
        return x.view(-1, 1, self.expand_to, self.expand_to)
    
    def forward(self, x):
        x = self.unflat(self.adaptation(x))
        return self.exit(self.backbone(x))  



'''
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
TryNet07                                 --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       200,704
│    └─Linear: 2-2                       16,781,312
│    └─ReLU: 2-3                         --
├─Sequential: 1-2                        --
│    └─Sequential: 2-4                   --
│    │    └─Conv2d: 3-1                  20,992
│    │    └─BatchNorm2d: 3-2             512
│    │    └─ReLU: 3-3                    --
│    │    └─MaxPool2d: 3-4               --
├─Sequential: 1-3                        --
│    └─Flatten: 2-5                      --
│    └─Linear: 2-6                       8,194
=================================================================
Total params: 17,011,714
Trainable params: 17,011,714
Non-trainable params: 0
=================================================================

Epoch:  4 Batch: 3121 Loss: 0.3224 Accuracy Train: 84.516%

Duration: 909 seconds

'''

class TryNet07(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.expand_to = 64

        self.adaptation = nn.Sequential(
            nn.Linear(48, self.expand_to ** 2),
            nn.Linear(4096, 4096),
            nn.ReLU())
        
        self.backbone = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=9, stride=2, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=9, stride=5)),
        )
        
        self.exit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, num_classes),
        )
    
    def unflat(self, x):
        return x.view(-1, 1, self.expand_to, self.expand_to)
    
    def forward(self, x):
        x = self.unflat(self.adaptation(x))
        return self.exit(self.backbone(x))  



'''
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
TryNet08                                 --
├─Sequential: 1-1                        --
│    └─Linear: 2-1                       50,176
│    └─ReLU: 2-2                         --
│    └─Linear: 2-3                       4,198,400
│    └─ReLU: 2-4                         --
│    └─Linear: 2-5                       4,195,328
│    └─ReLU: 2-6                         --
│    └─Linear: 2-7                       2,050
=================================================================
Total params: 8,445,954
Trainable params: 8,445,954
Non-trainable params: 0
=================================================================


Epoch:  0 Batch: 3111 Loss: 0.6390 Accuracy Train: 59.658%
Epoch:  0 Batch: 3121 Loss: 0.6540 Accuracy Train: 59.660%
Epoch:  1 Batch:   1 Loss: 0.6251 Accuracy Train: 62.333%
Epoch:  1 Batch:  11 Loss: 0.6563 Accuracy Train: 61.485%
Epoch:  1 Batch:  21 Loss: 0.7184 Accuracy Train: 61.349%
Epoch:  1 Batch:  31 Loss: 0.6553 Accuracy Train: 61.312%

Epoch:  3 Batch: 2351 Loss: 0.6931 Accuracy Train: 49.944%
Epoch:  3 Batch: 2361 Loss: 0.7025 Accuracy Train: 49.943%
Epoch:  3 Batch: 2371 Loss: 0.6931 Accuracy Train: 49.946%
'''
class TryNet08(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(48, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        return self.layers(x) 