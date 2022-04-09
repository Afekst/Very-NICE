import argparse
import torch
import torchvision
import utils
import nice
import verynice
import matplotlib.pyplot as plt
from tqdm import tqdm


def train(flow, data_loader, optimizer, device, epochs):
    loss_list = []
    for epoch in range(1, epochs+1):
        flow.train()
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
        generate(flow=flow, sample_size=64, sample_shape=[1, 28, 28], epoch=epoch)
        loss_list.append(running_loss/n_batch)
    return loss_list


def generate(flow, sample_size, sample_shape, epoch):
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(sample_size)
        samples = samples.view(samples.size(0), -1)
        # samples -= samples.min(1, keepdim=True)[0]
        # samples /= samples.max(1, keepdim=True)[0]
        samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
        samples = samples.cpu() + 0.5
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + 'samples ' + 'epoch %d.png' % epoch)


def plot_loss(loss_list):
    fig, ax = plt.subplots()
    ax.plot(loss_list)
    ax.set_title("LL Loss Vs. epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig('./loss/' + f"{args.dataset}_loss.png")


def main(args):
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = utils.retrieve_dataset(args.dataset, args.batch_size)

    # model = nice.NICE(
    #    prior=args.prior,
    #    coupling=args.coupling,
    #    in_out_dim=28**2,
    #    max_neurons=args.max_neurons,
    #    hidden=args.hidden,
    #    device=device
    # )
    model = verynice.VeryNICE(
        prior=args.prior,
        coupling=args.coupling,
        in_out_dim=28**2,
        max_neurons=args.max_neurons/10,
        partitions=8,
        hidden=args.hidden,
        device=device
    )
    utils.print_parameters(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train(model, train_loader, optimizer, device, args.epochs)
    plot_loss(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--cuda',
                        help='cuda device index',
                        type=int,
                        default=3)
    parser.add_argument('--dataset',
                        help='mnist or fashion-mnist',
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
                        default='logistic')
    parser.add_argument('--coupling',
                        help='number of coupling blocks',
                        type=int,
                        default=4)
    parser.add_argument('--max_neurons',
                        help='maximal number of hidden neurons',
                        type=int,
                        default=10e6)
    parser.add_argument('--hidden',
                        help='number of hidden layers in each coupling block',
                        type=int,
                        default=5)
    args = parser.parse_args()
    main(args)
