import torch, sys, os, h5py
sys.path.append('.')
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DatasetAll
from model.model import model
from utils import AverageValueMeter, Logger

random_seed = 1
torch.manual_seed(random_seed)
log_dir = 'log'
version = 'sample2'

epoches = 300
batch_size = 12

lr = 0.01
weight_decay = 1e-5
scheduler_step_size = 40
scheduler_gamma = 0.5

train_h5 = h5py.File('data/New_Train_data_400_400.h5', 'r')
train_data, train_label = train_h5['data'], train_h5['label']
train_dataset = DatasetAll(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size, True, num_workers=0)

test_h5 = h5py.File('data/New_Test_data_400_400.h5', 'r')
test_data, test_label = test_h5['data'], test_h5['label']
test_dataset = DatasetAll(test_data, test_label)
test_loader = DataLoader(test_dataset, batch_size, False, num_workers=0)

metrics = {'loss', 'acc'}
train_meter = {m: AverageValueMeter() for m in metrics}
test_meter = {m: AverageValueMeter() for m in metrics}
logger = Logger(f'{log_dir}/{version}')

os.system(f'cp -r model/model.py {log_dir}/{version}')


ce_loss = torch.nn.CrossEntropyLoss()
network = model(214, [128, 64, 1]).cuda()

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
    
    for batch, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
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
        # if (batch+1) % 25 == 0:
        logger.write('Train Epoch {} [{}/{}] Loss: {:.6f} Acc {:.6f}\n'.format(
                epoch, batch+1, len(train_loader), train_meter['loss'].avg, train_meter['acc'].avg))

    logger.write('Train Epoch {} Loss: {:.6f} Acc {:.6f}\n'.format(epoch, train_meter['loss'].avg, train_meter['acc'].avg))
    
    network.eval()
    for k,v in test_meter.items():
        test_meter[k].reset()
    
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

    logger.write('Test Loss {:.6f} Acc {:.6f}\n'.format(test_meter['loss'].avg, test_meter['acc'].avg))
    
    torch.save(network.state_dict(), f'{log_dir}/{version}/{epoch}_model.pth')
    # torch.save(optimizer.state_dict(), f'{log_dir}/{version}/{epoch}_optimizer.pth')

train_h5.close()
test_h5.close()