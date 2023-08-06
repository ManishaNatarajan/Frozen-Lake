import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import random
from data_loaders.BC_loader import BCDataset
from BC.model import BCModel


def train(device, train_dataloader, test_dataloader,
          batch_size, model, learning_rate, n_epochs, log_dir, use_actions=False):
    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=learning_rate)

    losses = []
    train_loss = 0
    best_test_loss = np.inf

    weight_regularization = 1
    # Initialize for writing on tensorboard
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(log_dir, str(time))

    # Create directories for saving models and tensorboard logs
    if not os.path.isdir(os.path.join(log_dir, 'models')):
        os.makedirs(os.path.join(log_dir, 'models'))

    # TODO: Dump training configs

    summary_dir = os.path.join(log_dir, str(time))
    writer = SummaryWriter(log_dir=summary_dir)

    i = 0
    model.writer = writer

    for epoch in tqdm(range(1, n_epochs + 1)):
        batch_train_loss = 0
        num_batches = 0

        for tup in train_dataloader:
            model.train()
            x_train, y_train = tup

            if use_actions:
                x1, x2 = x_train
                x_train = [x1.to(device).float(), x2.to(device).float()]
            else:
                x_train = x_train.to(device).float()
            y_train = y_train.to(device).float()

            i += 1
            num_batches += 1

            train_loss = model.compute_loss(x_train, y_train)
            batch_train_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train/neg_logp_train', train_loss.item(), i)

        losses.append(batch_train_loss)
        writer.add_scalar('loss/train/overall_loss', batch_train_loss / (num_batches), epoch)

        # After every n epochs evaluate
        if epoch % 2 == 0:
            # model.eval()  # Check with and without
            batch_test_loss = 0
            num_batches_test = 0
            accuracy = 0
            for tup_test in test_dataloader:
                x_test, y_test = tup_test

                if use_actions:
                    x1, x2 = x_test
                    x_test = [x1.to(device).float(), x2.to(device).float()]
                else:
                    x_test = x_test.to(device).float()
                y_test = y_test.to(device).float()

                num_batches_test += 1

                model_predictions = model.get_predictions(x_test)

                test_loss = F.cross_entropy(input=model_predictions, target=y_test)
                batch_test_loss += test_loss.item()

                # Compute accuracy...
                accuracy += torch.sum(torch.argmax(model_predictions, axis=-1) == torch.argmax(y_test, axis=-1))/(y_test.shape[0])
                # vals, prediction_counts = torch.argmax(model_predictions, axis=-1).unique(return_counts=True)
                # print(vals, prediction_counts)

            writer.add_scalar('accuracy/test/overall_accuracy', accuracy / (num_batches_test), epoch)
            writer.add_scalar('loss/test/overall_loss', batch_test_loss / (num_batches_test), epoch)

            if log_dir:
                if batch_test_loss < best_test_loss:
                    best_test_loss = batch_test_loss
                    print(f"Saving Best Model... {batch_test_loss / num_batches_test}")
                    torch.save(model.state_dict(), os.path.join(log_dir, "best.pth"))

            torch.save(model.state_dict(), os.path.join(log_dir, f"models/{epoch}.pth"))

            model.train()  # Switch back to train mode after eval


def main():
    # Set train parameters
    batch_size = 32
    seq_len = 16
    epochs = 100
    learning_rate = 1e-3
    use_actions = True

    num_human_actions = 5

    # Load datasets
    train_path = "data/User_Study_1/RL_data/train/"
    test_path = "data/User_Study_1/RL_data/val/"

    train_dataset = BCDataset(folder_path=train_path, sequence_length=seq_len, use_actions=use_actions,
                              num_human_actions=num_human_actions)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = BCDataset(folder_path=test_path, sequence_length=seq_len, use_actions=use_actions,
                             num_human_actions=num_human_actions)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = BCModel(obs_shape=(8, 8, 3), robot_action_shape=5, human_action_shape=num_human_actions,
                    conv_hidden=32, action_hidden=32, num_layers=1, use_actions=use_actions)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    log_dir = "BC_logs/"

    train(device=device, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
          batch_size=batch_size, model=model, learning_rate=learning_rate, n_epochs=epochs, log_dir=log_dir,
          use_actions=use_actions)


if __name__ == '__main__':
    seed = 0

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main()
