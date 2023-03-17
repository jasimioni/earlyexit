import torch
import torch.nn as nn

# CrossEntropyConfidence
# Custom Loss Function adding a weight according to softmax

class CrossEntropyConfidence(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, output, target):
        target = torch.tensor(target, dtype=torch.int64, device=self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)

        #print(f'For this batch: {output.shape} and {target.shape}')
        #print(f'{output[0]} and {target[0]}')
        cnf = torch.mean(torch.max(nn.functional.softmax(output, dim=-1), 1)[0]).item()
        loss = criterion(output, target)

        # print(f'CNF: {cnf} - Loss: {loss}')

        return loss - cnf


# show_exits_status
# Show the accuracy, timing and loss of the model for each exit, using the test datase

def show_exits_stats(model, test_loader, criterion=nn.CrossEntropyLoss(), device='cpu'):
    fast_inference_mode = model.fast_inference_mode
    measurement_mode = model.measurement_mode
    model.set_fast_inference_mode(False)
    model.set_measurement_mode(True)
    tst_cnt = 0
    tst_cor = [ 0 for exit in model.exits ]
    total_times = [ [ 0, 0 ] for exit in model.exits ]
    losses = [ 0 for exit in model.exits ]

    # model.eval()

    with torch.no_grad():
        # Run one batch to make sure the computations of first evaluation don't generate trouble        
        for (X_test, y_test) in test_loader:
            X_test = X_test.to(device)
            model(X_test)
            break

        for b, (X_test, y_test) in enumerate(test_loader):
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            y_val = model(X_test)

            for exit, y_val_exit in enumerate(y_val):
                predicted = torch.max(y_val_exit[0].data, 1)[1]
                batch_corr = (predicted == y_test).sum()
                tst_cor[exit] += batch_corr            

                total_times[exit][0] += y_val_exit[1]
                total_times[exit][1] += y_val_exit[2]

                losses[exit] += len(predicted) * criterion(y_val_exit[0], y_test).item()

            tst_cnt += len(predicted)
                                
    accs   = [ f'{100*tst_cor_exit/tst_cnt:2.2f}%' for tst_cor_exit in tst_cor ]
    times  = [ [ f'{1000*time[0]:.2f}ms', f'{1000*time[1]:.2f}ms' ] for time in total_times ]
    avg_losses = [ f'{loss/tst_cnt:2.2f}' for loss in losses ]
    
    print(f"\nTests: {tst_cnt} | Loss: {avg_losses} | Accuracy Test: {accs} | Times: {times}\n")
    model.set_fast_inference_mode(fast_inference_mode)
    model.set_measurement_mode(measurement_mode)

# train_exit:
# Train only one exit (only it's ouput is applied to .backward)
# backbone_parameters can be used.
# path: means all backbone until the exit is added in the parameters list of the optimizer
# section: means only the backbone section associated to the exit is added
# none: backbone parameters are not added to the parameters list
# 
# train_exit with last exit and backbone_parameters='path' is the same as pre-train backbone

def train_exit(model, exit, train_loader=None, test_loader=None, lr=0.001, epochs=5, 
               device='cpu', criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam, 
               backbone_parameters='path'):

    exit_params = []

    if backbone_parameters == 'path':
        for bb in model.backbone[0:exit+1]:
            exit_params.append({'params': bb.parameters()})
    elif backbone_parameters == 'section':
        exit_params.append({'params': model.backbone[exit].parameters()}) 

    exit_params.append({'params': model.exits[exit].parameters()})

    optimizer = optimizer(exit_params, lr=lr)

    print(f"Training exit {exit} with backbone_parameters={backbone_parameters} - Sections: {len(exit_params)}")

    import time
    start_time = time.time()

    for i in range(epochs):
        trn_cor = 0
        trn_cnt = 0
        tst_cor = 0
        tst_cnt = 0
        
        for b, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            b+=1
            
            y_pred = model(X_train)[exit] 
            loss = criterion(y_pred, y_train)
    
            predicted = torch.max(y_pred.data, 1)[1]
            batch_cor = (predicted == y_train).sum()
            trn_cor += batch_cor
            trn_cnt += len(predicted)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (b-1)%10 == 0:
                print(f'Epoch: {i:2} Batch: {b:3} Loss: {loss.item():4.4f} Accuracy Train: {trn_cor.item()*100/trn_cnt:2.3f}%')
            
        show_exits_stats(model, test_loader, criterion, device)
            
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')


# train_model:
# This function trains all exits at once. It does that by applying .backward to each exit value
# The .backward is applyed one exit at a time, beginning with the nearest one
# Loss value from each exit is multiplied by a weight factor defined in the model (exit_loss_weights)

def train_model(model, train_loader=None, test_loader=None, lr=0.001, epochs=5, 
                device='cpu', criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):

    optimizer = optimizer(model.parameters(), lr=lr)

    import time
    start_time = time.time()

    for i in range(epochs):
        trn_cor = [0, 0, 0]
        trn_cnt = 0
        
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            b+=1
            
            y_pred = model(X_train)  
                
            losses = [weighting * criterion(res, y_train) for weighting, res in zip(model.exit_loss_weights, y_pred)]
            
            optimizer.zero_grad()        
            for loss in losses[:-1]:
                loss.backward(retain_graph=True)
            losses[-1].backward()
            optimizer.step()
            
            for exit, y_pred_exit in enumerate(y_pred):   
                predicted = torch.max(y_pred_exit.data, 1)[1]
                batch_corr = (predicted == y_train).sum()
                trn_cor[exit] += batch_corr
                    
            trn_cnt += len(predicted)
            
            if (b-1)%10 == 0:
                loss_string = [ f'{loss.item():4.4f}' for loss in losses ]
                accu_string = [ f'{correct.item()*100/trn_cnt:2.3}%' for correct in trn_cor ]
                print(f'Epoch: {i:2} Batch: {b:3} Loss: {loss_string} Accuracy Train: {accu_string}%')
            
        show_exits_stats(model, test_loader, criterion, device)
            
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
