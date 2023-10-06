import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import argparse
import math
from torchvision import datasets, transforms
import time
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
import os
import pandas as pd

# NT parametrization, including learning rate and standard deviation of initialization.
class MLP(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=3072, out_dim=1, nonlin=F.relu):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        self.fc_1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc_3 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc_1.weight, std=1. / math.sqrt(self.fc_1.weight.size(1)))
        nn.init.normal_(self.fc_2.weight, std=1. / math.sqrt(self.fc_2.weight.size(1)))
        nn.init.normal_(self.fc_3.weight, std=1. / math.sqrt(self.fc_3.weight.size(1)))

    def forward(self, x):
        out = self.nonlin(self.fc_1(x))
        out = self.nonlin(self.fc_2(out))
        out = self.fc_3(out)
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''pytorch for implement mlp''',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--output_mult', type=float, default=1.0)
    parser.add_argument('--input_mult', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=300)
    parser.add_argument('--log_dir', type=str, default='.')
    parser.add_argument('--tensorboard', action='store_true',
                        help='store the training information')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='shuffle training data')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    def post_process(data):
        unit_x = []
        unit_y = []
        for unit in data:
            unit_x.append(unit[0])
            label = 2 * (unit[1] % 2 == 0) - 1
            unit_y.append(label)
        out_x = torch.stack(unit_x, dim=0)
        out_y = torch.tensor(unit_y, dtype=torch.float32)
        out_y = out_y.view(-1, 1)
        return out_x, out_y


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = \
        datasets.CIFAR10(root=args.data_dir, train=True,
                         download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=not args.no_shuffle, num_workers=2, collate_fn=post_process)

    testset = datasets.CIFAR10(root=args.data_dir, train=False,
                               download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=2, collate_fn=post_process)


    def train(args, model, device, train_loader, optimizer, epoch,
              scheduler=None, criterion=F.mse_loss):
        model.train()
        train_loss = 0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item() * data.shape[0]  # sum up batch loss
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                elapsed = time.time() - start_time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | ms/batch {:5.2f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           elapsed * 1000 / args.log_interval))
                start_time = time.time()
            if scheduler is not None:
                scheduler.step()
        train_loss /= len(train_loader.dataset)
        return train_loss


    def test(args, model, device, test_loader,
             evalmode=True, criterion=F.mse_loss):
        if evalmode:
            model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data.view(data.size(0), -1))
                test_loss += criterion(output, target).item() * data.shape[0]

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        return test_loss


    logs = []
    for width in [512]:  # 256, 512, 1024, 2048, 4096, 8192
        for lr in np.logspace(-4, 0., 10):  # iterate learning rate, np.logspace(-12, -4, 9, base=2)
            if args.tensorboard:
                writer = SummaryWriter("./runs/" + f'{str(width)}_lr_{lr:.5f}')
            mynet = MLP(hidden_dim=width, out_dim=1, nonlin=F.relu).to(device)
            optimizer = SGD([{'params': n.parameters(), "lr":float(lr/n.weight.size(1))} for n in mynet.modules() if isinstance(n, nn.Linear)])  # customized lr
            for epoch in range(1, args.epochs + 1):
                train_loss = train(args, mynet, device, train_loader, optimizer, epoch,
                                   criterion=F.mse_loss)
                test_loss = test(args, mynet, device, test_loader, criterion=F.mse_loss)
                norms = {}
                for layer, para in enumerate(mynet.parameters()):
                    with torch.no_grad():
                        norms[f"layer{layer}"] = torch.norm(para.data, p="fro")
                if args.tensorboard:
                    writer.add_scalars(f"loss",
                                       {"train_loss": train_loss, "test_loss": test_loss}, epoch)
                    writer.add_scalars(f"norm", norms, epoch)
                logs.append(dict(
                    epoch=epoch,
                    train_loss=train_loss,
                    test_loss=test_loss,
                    width=width,
                    lr=round(lr, 5),
                ))
                if math.isnan(train_loss):
                    break
            if args.tensorboard:
                writer.close()

    with open(os.path.join(os.path.expanduser(args.log_dir), 'logs_experiment.tsv'), 'w') as f:
        logdf = pd.DataFrame(logs)
        print(os.path.join(os.path.expanduser(args.log_dir), 'logs_experiment.tsv'))
        f.write(logdf.to_csv(sep='\t', float_format='%.4f'))
