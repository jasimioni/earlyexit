import torch
import torch.nn as nn
import time

class DynNetFlat(nn.Module):
    def __init__(self, input_sample=None, num_classes=2):
        super().__init__()

        x, y = input_sample[0]

        c, = x.shape

        self.layers = nn.Sequential(
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(c, 25)),

            nn.Sequential(
                nn.ReLU(),
                nn.Linear(25, num_classes)),
        )
    
    def forward(self, x):
        return self.layers(x) 


class DynNetS(nn.Module):
    def __init__(self, input_sample=None, num_classes=2):
        super().__init__()

        x, y = input_sample[0]

        x = x.view(1, *x.shape)
        c, l, w, h = x.shape
        flat_size = l * w * h
        
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(flat_size, num_classes))
        )
    
    def forward(self, x):
        return self.layers(x) 

class DynNetGen(nn.Module):
    def __init__(self, 
                 input_sample=None, 
                 num_classes=2, 
                 conv_filters=None, 
                 conv_kernel_sizes=None,
                 conv_strides=None,
                 conv_paddings=None):

        super().__init__()

        if input_sample is None:
            raise Exception("No sample data provided - must be in the format of (1, c, r)")

        x, y = input_sample[0]
        x = x.view(1, *x.shape)
        self.layers = nn.Sequential()

        if conv_filters is not None:
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [ 2 for i in conv_filters ]
            if conv_strides is None:
                conv_strides = [ 1 for i in conv_filters ]
            if conv_paddings is None:
                conv_paddings = [ 0 for i in conv_filters ]

            l_filter_size = 1
            for i, filter_size in enumerate(conv_filters):
                self.layers.append(nn.Sequential(
                    nn.Conv2d(l_filter_size, 
                              filter_size, 
                              kernel_size=conv_kernel_sizes[i], 
                              stride=conv_strides[i], 
                              padding=conv_paddings[i]),
                    nn.ReLU()))
                l_filter_size = filter_size
                    
            x = self.layers(x)
        
        c, l, w, h = x.shape
        
        flat_size = l * w * h
        
        self.layers.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, num_classes))
        )
    
    def forward(self, x):
        return self.layers(x) 

class DynNetM(nn.Module):
    def __init__(self, input_sample=None, num_classes=2):
        super().__init__()
        
        x, y = input_sample[0]

        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU()),        
        )

        x = x.view(1, *x.shape)
        x = self.layers(x)
        
        c, l, w, h = x.shape
        
        flat_size = l * w * h
        
        self.layers.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, num_classes))
        )
    
    def forward(self, x):
        return self.layers(x) 

class DynNetL(nn.Module):
    def __init__(self, input_sample=None, num_classes=2):
        super().__init__()

        x, y = input_sample[0]

        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU()),        
            
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU()),     
        )

        x = x.view(1, *x.shape)
        x = self.layers(x)
        
        c, l, w, h = x.shape
        
        flat_size = l * w * h
        
        self.layers.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, num_classes))
        )
                    
    
    def forward(self, x):
        return self.layers(x) 

class MooreNetLayers():
    def __init__(self, num_classes=2):
        self.backbone = [
            nn.Sequential(),
            
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(32),
                    nn.ReLU()),        

            ),
            
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
                    nn.BatchNorm2d(64),
                    nn.ReLU()),
                
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
                    nn.BatchNorm2d(128),
                    nn.ReLU()),
            ),
        ]

        self.exits = [
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, num_classes)),
            
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1600, num_classes)),

            nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, num_classes))
        ]

class MooreNetWithExits(nn.Module):
    def __init__(self, num_classes=2, exit_loss_weights=[1.0, 0.5, 0.2]):
        super().__init__()

        layers = MooreNetLayers(num_classes=num_classes)

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