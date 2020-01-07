import time
from multiprocessing import cpu_count
from typing import Union

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import UrbanSound8KDataset

import argparse
from pathlib import Path
from cnn_model import CNN
from validate import validate_single

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train CNN for environment sound classification",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--sgd-momentum",
    default=0.9,
    type=float,
    help='Momentum value for the SGD optimizer.'
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--dropout",
    default=0.5,
    type=float,
    help="Probability of dropped neurons 0-1",
)
parser.add_argument(
    "--weight-decay",
    default=1e-5,
    type=float,
    help="Weight decay: parameter related to L-2 regularisation.",
)
parser.add_argument(
    "--checkpoint-path",
    default=Path("checkpoint.pkl"),
    type=Path,
    help="Provide a file to store checkpoints of the model parameters during training."
)
parser.add_argument(
    "--checkpoint-frequency",
    type=int, default=5,
    help="Save a checkpoint every N epochs"
)
parser.add_argument(
    "--mode",
    type=str, default='LMC',
    help="Mode of network to train."
)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    transform = transforms.ToTensor()
    mode = args.mode
    train_data = UrbanSound8KDataset('UrbanSound8K_train.pkl', mode)
    print(f"train length: [{len(train_data)}]")
    train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.worker_count,
      pin_memory=True
    )
    test_data = UrbanSound8KDataset('UrbanSound8K_test.pkl', mode)
    print(f"test length: [{len(test_data)}]")
    val_loader = torch.utils.data.DataLoader(
     test_data,
     batch_size=args.batch_size,
     shuffle=True,
     num_workers=args.worker_count,
     pin_memory=True
    )

    model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout,mode = args.mode)

    ## TASK 8: Redefine the criterion to be softmax cross entropy
    criterion = nn.CrossEntropyLoss()

    ## Use adam optimizer. AdamW is Adam with L-2 regularisation.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ##optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum = 0.9, weight_decay=args.weight_decay)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer,
        DEVICE, args.checkpoint_path, checkpoint_frequency = args.checkpoint_frequency
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        checkpoint_path: Path,
        checkpoint_frequency: int = 5
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency


    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for i, (inputs, targets, filenames) in enumerate(self.train_loader):
            # for batch, labels in self.train_loader:
                batch = inputs.to(self.device)
                labels = targets.to(self.device)
                data_load_end_time = time.time()

                ## Forward pass through the network.
                logits = self.model.forward(batch)

                ## Format logits
                results = {"preds": [], "labels": []}
                file_logits = torch.tensor([]).to(self.device)
                fname_to_index = {}
                # for each sample in this batch:
                for k in range(len(filenames)):
                    fname = filenames[k]
                    if fname in fname_to_index.keys():
                        # add the logits of the same file together.
                        file_logits[fname_to_index[fname]] += logits[k]
                    else:
                        # store the sumation of the logits and label for this file.
                        fname_to_index[fname] = len(file_logits)
                        file_logits = torch.cat((file_logits,logits[k:k+1]),0)
                        results["labels"].append(labels.cpu().numpy()[k])

                ## Compute the loss using the specified criterion.
                loss = self.criterion(file_logits, results["labels"])

                ## Compute the backward pass
                loss.backward()

                ## Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = file_logits.argmax(-1)
                    results["preds"].extend(list(preds))
                    accuracy = compute_accuracy(results["labels"], results["preds"])

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)
                self.step += 1
                data_load_start_time = time.time()

            self.model_checkpoint(accuracy, epoch, epochs)
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):

        average_loss, accuracy, class_accuracies = validate_single(self.model, self.val_loader, self.criterion, self.device)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

        print(f"ac: {class_accuracies[0].item() * 100:2.2f}")
        print(f"ch: {class_accuracies[1].item() * 100:2.2f}")
        print(f"cp: {class_accuracies[2].item() * 100:2.2f}")
        print(f"db: {class_accuracies[3].item() * 100:2.2f}")
        print(f"dr: {class_accuracies[4].item() * 100:2.2f}")
        print(f"ei: {class_accuracies[5].item() * 100:2.2f}")
        print(f"gs: {class_accuracies[6].item() * 100:2.2f}")
        print(f"jh: {class_accuracies[7].item() * 100:2.2f}")
        print(f"si: {class_accuracies[8].item() * 100:2.2f}")
        print(f"sm: {class_accuracies[9].item() * 100:2.2f}")

    def model_checkpoint(self, accuracy, epoch, epochs):
        if (epoch + 1) % self.checkpoint_frequency == 0 or (epoch + 1) == epochs:
            print(f"Saving model to {self.checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'accuracy': accuracy
            }, self.checkpoint_path)


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

def per_class_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """

    class_accuracies = torch.zeros(10,)
    for i in range(10):
        idx = labels == i
        class_accuracies[i] = float((labels[idx] == preds[idx]).sum()) / idx.sum()

    return class_accuracies

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_wd={args.weight_decay}_VERSION'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
