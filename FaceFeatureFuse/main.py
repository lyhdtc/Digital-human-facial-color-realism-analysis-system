import torch, sys, os
sys.path.append('.')
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import Dataset
from model.model import model
from utils import AverageValueMeter, Logger

random_seed = 1
torch.manual_seed(random_seed)
log_dir = 'log'
version = '328'

epoches = 300
_range = 12
batch_size = 6

lr = 0.01
weight_decay = 1e-5
scheduler_step_size = 40
scheduler_gamma = 0.5

metrics = {'loss', 'acc'}
train_meter = {m: AverageValueMeter() for m in metrics}
test_meter = {m: AverageValueMeter() for m in metrics}
logger = Logger(f'{log_dir}/{version}')

ce_loss = torch.nn.CrossEntropyLoss()
network = model(214, [64, 24, 1]).cuda()

optimizer = torch.optim.AdamW(
            network.parameters(), 
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay = weight_decay,
        )

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, scheduler_gamma)
# train
best_acc = 0.
for epoch in range(epoches):
    network.train()
    for k,v in train_meter.items():
        train_meter[k].reset()
    
 
    _start, _end = 0, 0
    for i in tqdm(range(700//_range), total=700//_range):
        _end += _range
        train_dataset = Dataset('data/Train_data.h5', start=_start, end=_end)
        _start = _end

        train_loader = DataLoader(train_dataset, batch_size, True, num_workers=0)
        for batch, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output, _ = network(data)
        
            loss = ce_loss(output, target[:, 0])
            loss.backward()
            optimizer.step()

            pred = output.data.max(1, keepdim=True)[1]
            acc = pred.eq(target.data.view_as(pred)).sum() / float(batch_size)

            train_meter['loss'].update(loss.item())
            train_meter['acc'].update(acc)

            logger.write('Train Epoch {} [{}/{}] Loss: {:.6f} Acc {:.6f}\n'.format(
                    epoch, i, 700//_range, train_meter['loss'].avg, train_meter['acc'].avg))

        del train_dataset, train_loader
    logger.write('Train Epoch {} Loss: {:.6f} Acc {:.6f}\n'.format(epoch, train_meter['loss'].avg, train_meter['acc'].avg))
    
    network.eval()
    for k,v in test_meter.items():
        test_meter[k].reset()
    
    _start, _end = 0, 0
    for i in range(300//_range):
        _end += _range
        test_dataset = Dataset('data/Test_data.h5', start=0, end=_end)
        _start = _end

        test_loader = DataLoader(test_dataset, batch_size, False, num_workers=0)

        with torch.no_grad():
            for data, target in tqdm(test_loader, total=len(test_loader), smoothing=0.9):
                data = data.cuda()
                target = target.cuda()
                output, _ = network(data)
                test_loss = ce_loss(output, target[:, 0]).item()
                pred = output.data.max(1, keepdim=True)[1]
                acc = pred.eq(target.data.view_as(pred)).sum() / float(batch_size)

                test_meter['loss'].update(loss.item())
                test_meter['acc'].update(acc.item())


        del test_dataset, test_loader
    
    logger('Test Loss {:.6f} Acc {:.6f}\n'.format(test_meter['loss'].avg, test_meter['acc'].avg))
    if acc > best_acc:
        best_acc = acc
        torch.save(network.state_dict(), f'{log_dir}/model.pth')
        torch.save(optimizer.state_dict(), f'{log_dir}/optimizer.pth')

    