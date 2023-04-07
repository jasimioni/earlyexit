```
===========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
===========================================================================
JoaoNetWithExitsCIFAR10                  [1, 10]                   --
├─Sequential: 1-5                        --                        (recursive)
│    └─Sequential: 2-1                   [1, 96, 14, 14]           --
│    │    └─Conv2d: 3-1                  [1, 96, 30, 30]           2,688
│    │    └─BatchNorm2d: 3-2             [1, 96, 30, 30]           192
│    │    └─ReLU: 3-3                    [1, 96, 30, 30]           --
│    │    └─MaxPool2d: 3-4               [1, 96, 14, 14]           --
├─Sequential: 1-6                        --                        (recursive)
│    └─Sequential: 2-2                   [1, 10]                   --
│    │    └─MaxPool2d: 3-5               [1, 96, 5, 5]             --
│    │    └─Flatten: 3-6                 [1, 2400]                 --
│    │    └─Linear: 3-7                  [1, 10]                   24,010
├─Sequential: 1-5                        --                        (recursive)
│    └─Sequential: 2-3                   [1, 256, 7, 7]            --
│    │    └─Conv2d: 3-8                  [1, 256, 16, 16]          221,440
│    │    └─BatchNorm2d: 3-9             [1, 256, 16, 16]          512
│    │    └─ReLU: 3-10                   [1, 256, 16, 16]          --
│    │    └─MaxPool2d: 3-11              [1, 256, 7, 7]            --
├─Sequential: 1-6                        --                        (recursive)
│    └─Sequential: 2-4                   [1, 10]                   --
│    │    └─MaxPool2d: 3-12              [1, 256, 2, 2]            --
│    │    └─Flatten: 3-13                [1, 1024]                 --
│    │    └─Linear: 3-14                 [1, 10]                   10,250
├─Sequential: 1-5                        --                        (recursive)
│    └─Sequential: 2-5                   [1, 256, 7, 7]            --
│    │    └─Conv2d: 3-15                 [1, 384, 7, 7]            885,120
│    │    └─BatchNorm2d: 3-16            [1, 384, 7, 7]            768
│    │    └─ReLU: 3-17                   [1, 384, 7, 7]            --
│    │    └─Conv2d: 3-18                 [1, 256, 7, 7]            884,992
│    │    └─BatchNorm2d: 3-19            [1, 256, 7, 7]            512
│    │    └─ReLU: 3-20                   [1, 256, 7, 7]            --
├─Sequential: 1-6                        --                        (recursive)
│    └─Sequential: 2-6                   [1, 10]                   --
│    │    └─MaxPool2d: 3-21              [1, 256, 2, 2]            --
│    │    └─Flatten: 3-22                [1, 1024]                 --
│    │    └─Linear: 3-23                 [1, 10]                   10,250
===========================================================================
Total params: 2,040,734
Trainable params: 2,040,734
Non-trainable params: 0
Total mult-adds (M): 145.89
===========================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 2.93
Params size (MB): 8.16
Estimated Total Size (MB): 11.11
===========================================================================
```

```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
JoaoNetWithExitsCIFAR10                  --
├─Sequential: 1-1                        --
│    └─Sequential: 2-1                   --
│    │    └─Conv2d: 3-1                  2,688
│    │    └─BatchNorm2d: 3-2             192
│    │    └─ReLU: 3-3                    --
│    │    └─MaxPool2d: 3-4               --
│    └─Sequential: 2-2                   --
│    │    └─Conv2d: 3-5                  221,440
│    │    └─BatchNorm2d: 3-6             512
│    │    └─ReLU: 3-7                    --
│    │    └─MaxPool2d: 3-8               --
│    └─Sequential: 2-3                   --
│    │    └─Conv2d: 3-9                  885,120
│    │    └─BatchNorm2d: 3-10            768
│    │    └─ReLU: 3-11                   --
│    │    └─Conv2d: 3-12                 884,992
│    │    └─BatchNorm2d: 3-13            512
│    │    └─ReLU: 3-14                   --
├─Sequential: 1-2                        --
│    └─Sequential: 2-4                   --
│    │    └─MaxPool2d: 3-15              --
│    │    └─Flatten: 3-16                --
│    │    └─Linear: 3-17                 24,010
│    └─Sequential: 2-5                   --
│    │    └─MaxPool2d: 3-18              --
│    │    └─Flatten: 3-19                --
│    │    └─Linear: 3-20                 10,250
│    └─Sequential: 2-6                   --
│    │    └─MaxPool2d: 3-21              --
│    │    └─Flatten: 3-22                --
│    │    └─Linear: 3-23                 10,250
=================================================================
Total params: 2,040,734
Trainable params: 2,040,734
Non-trainable params: 0
=================================================================
```

```
JoaoNetWithExitsCIFAR10(
  (backbone): Sequential(
    (0): Sequential(
      (0): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (1): Sequential(
      (0): Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (2): Sequential(
      (0): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
    )
  )
  (exits): Sequential(
    (0): Sequential(
      (0): MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): Flatten(start_dim=1, end_dim=-1)
      (2): Linear(in_features=2400, out_features=10, bias=True)
    )
    (1): Sequential(
      (0): MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): Flatten(start_dim=1, end_dim=-1)
      (2): Linear(in_features=1024, out_features=10, bias=True)
    )
    (2): Sequential(
      (0): MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): Flatten(start_dim=1, end_dim=-1)
      (2): Linear(in_features=1024, out_features=10, bias=True)
    )
  )
)
```
