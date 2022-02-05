import argparse
import torch
import utils
import nice
from tqdm import tqdm, trange


def train(flow, data_loader, optimizer, device, epochs):
    flow.train()
    loss_list = []
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        with tqdm(data_loader) as tepoch:
            tepoch.set_description(f'Epoch[{epoch}/{epochs}]')
            tepoch.set_postfix(loss=float('inf'))
            for n_batch, data in enumerate(tepoch, 1):
                optimizer.zero_grad()
                data, _ = data
                data = data.to(device)
                data = data.view(data.shape[0], data.shape[1] * data.shape[2] * data.shape[3])
                loss = -flow(data).mean()
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        loss_list.append(running_loss/n_batch)


def main(args):
    device = torch.device(f'cuda{args.cuda}' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = utils.retrieve_dataset(args.dataset, args.batch_size)

    model = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        in_out_dim=28**2,
        mid_dim=args.mid_dim,
        hidden=args.hidden
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    print(train(model, train_loader, optimizer, device, args.epochs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--cuda',
                        help='cuda device index',
                        type=int,
                        default=0)
    parser.add_argument('--dataset',
                        help='mnist of fashion-mnist',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='size of batch',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='number of epochs',
                        type=int,
                        default=50)
    parser.add_argument('--prior',
                        help='logistic or gaussian',
                        type=str,
                        default='gaussian')
    parser.add_argument('--coupling',
                        help='number of coupling blocks',
                        type=int,
                        default=4)
    parser.add_argument('--mid_dim',
                        help='number of units in each hidden layer of the coupling block',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='number of hidden layers in each coupling block',
                        type=int,
                        default=5)
    args = parser.parse_args()
    main(args)
