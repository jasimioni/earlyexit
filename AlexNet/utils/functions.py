import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
writer = None
stats_seq_cnt = 0


def set_writer(path):
    global writer
    writer = SummaryWriter(log_dir=path)

# CrossEntropyConfidence:
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

        adjusted_loss = loss + loss * (1 - cnf)

        print(f'CNF: {cnf} - Loss: {loss} - Adjusted Loss: {adjusted_loss}')

        return adjusted_loss

# show_exits_stats:
# Show the accuracy, timing and loss of the model for each exit, using the test datase

def show_exits_stats(model, test_loader, criterion=nn.CrossEntropyLoss(), device='cpu'):
    global stats_seq_cnt

    fast_inference_mode = model.fast_inference_mode
    measurement_mode = model.measurement_mode
    model.set_fast_inference_mode(False)
    model.set_measurement_mode(True)
    tst_cnt = 0
    tst_cor = [ 0 for exit in model.exits ]
    total_times = [ [ 0, 0 ] for exit in model.exits ]
    losses = [ 0 for exit in model.exits ]
    cnfs = [ 0 for exit in model.exits ]

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

                cnfs[exit] += len(predicted) * torch.mean(torch.max(nn.functional.softmax(y_val_exit[0], dim=-1), 1)[0]).item()
                losses[exit] += len(predicted) * criterion(y_val_exit[0], y_test).item()

            tst_cnt += len(predicted)
                                
    accs   = [ 100*tst_cor_exit/tst_cnt for tst_cor_exit in tst_cor ]
    times  = [ [ f'{1000*time[0]:.2f}ms', f'{1000*time[1]:.2f}ms' ] for time in total_times ]
    avg_losses = [ loss/tst_cnt for loss in losses ]
    avg_cnfs = [ f'{cnf/tst_cnt:2.2f}' for cnf in cnfs ]

    for exit, acc in enumerate(accs):
        writer.add_scalar(f'Accuracy/test exit {exit}', acc, stats_seq_cnt)
    
    for exit, loss in enumerate(avg_losses):
        writer.add_scalar(f'Loss/test exit {exit}', loss, stats_seq_cnt)

    for exit, cnf in enumerate(cnfs):
        writer.add_scalar(f'CNF/test exit {exit}', cnf/tst_cnt, stats_seq_cnt)

    stats_seq_cnt += 1

    loss = ' '.join(f'{loss:2.2f}' for loss in avg_losses)
    acc = ' '.join(f'{acc:2.2f}' for acc in accs)

    print(f"\nTests: {tst_cnt} | Loss: {loss} | Accuracy Test: {acc} | Times: {times} | CNFs: {avg_cnfs}\n")
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

    seq = 0
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

            writer.add_scalar(f"Loss/train exit {exit}", loss, seq)
    
            predicted = torch.max(y_pred.data, 1)[1]
            batch_cor = (predicted == y_train).sum()
            trn_cor += batch_cor
            trn_cnt += len(predicted)

            cnf = torch.mean(torch.max(nn.functional.softmax(y_pred, dim=-1), 1)[0]).item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (b-1)%10 == 0:
                print(f'Epoch: {i:2} Batch: {b:3} Loss: {loss.item():4.4f} Accuracy Train: {trn_cor.item()*100/trn_cnt:2.3f}%')
            
            writer.add_scalar(f"Accuracy/train exit {exit}", trn_cor.item()*100/trn_cnt, seq)
            writer.add_scalar(f"CNF/train exit {exit}", cnf, seq)
            seq += 1

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

    seq = 0
    for i in range(epochs):
        trn_cor = [0, 0, 0]
        cnf = [0, 0, 0]
        trn_cnt = 0
        
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            b+=1
            
            y_pred = model(X_train)  
                
            losses = [weighting * criterion(res, y_train) for weighting, res in zip(model.exit_loss_weights, y_pred)]

            for exit, loss in enumerate(losses):
                writer.add_scalar(f"Loss/train multiple exit {exit}", loss, seq)

            optimizer.zero_grad()        
            for loss in losses[:-1]:
                loss.backward(retain_graph=True)
            losses[-1].backward()
            optimizer.step()
            
            for exit, y_pred_exit in enumerate(y_pred):   
                predicted = torch.max(y_pred_exit.data, 1)[1]
                cnf = torch.mean(torch.max(nn.functional.softmax(y_pred_exit, dim=-1), 1)[0]).item()
                writer.add_scalar(f"CNF/train multiple exit {exit}", cnf, seq)
                batch_corr = (predicted == y_train).sum()
                trn_cor[exit] += batch_corr
                    
            trn_cnt += len(predicted)

            for exit, correct in enumerate(trn_cor):
                writer.add_scalar(f"Accuracy/train multiple exit {exit}", correct.item()*100/trn_cnt, seq)
            
            if (b-1)%10 == 0:
                loss_string = [ f'{loss.item():4.4f}' for loss in losses ]
                accu_string = [ f'{correct.item()*100/trn_cnt:2.3}%' for correct in trn_cor ]
                print(f'Epoch: {i:2} Batch: {b:3} Loss: {loss_string} Accuracy Train: {accu_string}%')

            seq += 1
        
        show_exits_stats(model, test_loader, criterion, device)
            
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
