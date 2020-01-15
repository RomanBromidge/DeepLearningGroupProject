import argparse, torch
import numpy as np
from pathlib import Path
from cnn_model import CNN
from typing import Union
from dataset import UrbanSound8KDataset
from multiprocessing import cpu_count
from torch.nn import CrossEntropyLoss, Softmax

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Produce the validation accuracy of a trained model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--mode",
    default="LMC",
    type=str,
    help="Kind of model to evaluate. One of LMC, MC, MLMC or TSCNN."
)
parser.add_argument(
    "--checkpoint",
    default=Path("checkpoint.pkl"),
    type=Path,
    help="Path to the file with the parameters of the trained model."
)
parser.add_argument(
    "--checkpoint2",
    default=Path("checkpoint2.pkl"),
    type=Path,
    help="Path to the file with the parameters of the trained MC network for the fusion."
)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)

def validate_model(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    criterion = CrossEntropyLoss()
    mode = args.mode

    if mode in ['MC','LMC','MLMC']:
        validation_loader = torch.utils.data.DataLoader(
            UrbanSound8KDataset('UrbanSound8K_test.pkl', mode),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.worker_count,
            pin_memory=True
        )
        model = CNN(height=85, width=41, channels=1, class_count=10, dropout=0.5, mode = mode)
        checkpoint = torch.load(args.checkpoint, map_location = device)

        model.load_state_dict(checkpoint['model'])
        print(f"Validating {mode} model with parameters trained for {checkpoint['epoch']} epochs.")

        loss, accuracy, class_accuracies = validate_single(model, validation_loader, criterion, device)

        print(f"accuracy: {accuracy * 100:2.2f}")
        print_class_accuracies(class_accuracies)

    elif mode == 'TSCNN':
        loader_LMC = torch.utils.data.DataLoader(
            UrbanSound8KDataset('UrbanSound8K_test.pkl', 'LMC'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.worker_count,
            pin_memory=True
        )
        loader_MC = torch.utils.data.DataLoader(
            UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MC'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.worker_count,
            pin_memory=True
        )
        model1 = CNN(height=85, width=41, channels=1, class_count=10, dropout=0.5, mode = 'LMC')
        model2 = CNN(height=85, width=41, channels=1, class_count=10, dropout=0.5, mode = 'MC')

        checkpoint1 = torch.load(args.checkpoint, map_location = device)
        model1.load_state_dict(checkpoint1['model'])

        checkpoint2 = torch.load(args.checkpoint2, map_location = device)
        model2.load_state_dict(checkpoint2['model'])

        print(f"Validating {mode} model with parameters trained for {checkpoint1['epoch']} and {checkpoint2['epoch']} epochs.")
        accuracy, class_accuracies = validate_double(model1, model2, loader_LMC, loader_MC, criterion, device)

        print(f"accuracy: {accuracy * 100:2.2f}")
        print_class_accuracies(class_accuracies)
    else:
        print('Please provide a valid argument.')


def validate_single(model, validation_loader, criterion, device):
    results = {"preds": [], "labels": []}
    file_logits = torch.tensor([]).to(device)
    fname_to_index = {}
    total_loss = 0
    model.eval()
    model.float()

    with torch.no_grad():
        for i, (inputs, targets, filenames) in enumerate(validation_loader):
            batch = inputs.to(device)
            labels = targets.to(device)
            logits = model(batch.float())
            loss = criterion(logits, labels)
            total_loss += loss.item()
            # For each sample in this batch:
            for k in range(len(filenames)):
                fname = filenames[k]
                if fname in fname_to_index.keys():
                    # Add the logits of the same file together.
                    file_logits[fname_to_index[fname]] += logits[k]
                else:
                    # Store the sumation of the logits and label for this file.
                    fname_to_index[fname] = len(file_logits)
                    file_logits = torch.cat((file_logits,logits[k:k+1]),0)
                    results["labels"].append(labels.cpu().numpy()[k])
        preds = file_logits.argmax(dim=-1).cpu().numpy()
        results["preds"].extend(list(preds))

    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )

    print(f"Saving model to output-MLMC.pkl")
    output = np.equal(np.array(results["labels"]), np.array(results["preds"]))
    torch.save({
        'output': output
    }, "output-MLMC.pkl")

    average_loss = total_loss / len(validation_loader)

    class_accuracies = per_class_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    return average_loss, accuracy, class_accuracies

def validate_double(model1, model2, loader_LMC, loader_MC, criterion, device):
    results = {"preds": [], "labels": []}
    file_logits = torch.tensor([]).to(device)
    fname_to_index = {}
    total_loss = 0
    model1.eval()
    model2.eval()
    normalize = Softmax(dim=1)
    loader_MC = iter(loader_MC)
    model1.to(device)
    model2.to(device)

    with torch.no_grad():
        for i, (inputs1, targets, filenames) in enumerate(loader_LMC):
            (inputs2, targets2, _) = next(loader_MC)
            batch1 = inputs1.to(device)
            batch2 = inputs2.to(device)
            labels = targets.to(device)
            logits1 = normalize(model1(batch1))
            logits2 = normalize(model2(batch2))
            logits = (logits1 + logits2)/2
            loss = criterion(logits, labels)
            total_loss += loss.item()
            # For each sample in this batch:
            for k in range(len(filenames)):
                fname = filenames[k]
                if fname in fname_to_index.keys():
                    # Add the logits of the same file together.
                    file_logits[fname_to_index[fname]] += logits[k]
                else:
                    # Store the sumation of the logits and label for this file.
                    fname_to_index[fname] = len(file_logits)
                    file_logits = torch.cat((file_logits,logits[k:k+1]),0)
                    results["labels"].append(labels.cpu().numpy()[k])
        preds = file_logits.argmax(dim=-1).cpu().numpy()
        results["preds"].extend(list(preds))

    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )

    print(f"Saving model to output-TSCNN.pkl")
    output = np.equal(np.array(results["labels"]), np.array(results["preds"]))
    torch.save({
        'output': output
    }, "output-TSCNN.pkl")

    average_loss = total_loss / len(loader_LMC)

    class_accuracies = per_class_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    return accuracy, class_accuracies

def print_class_accuracies(class_accuracies):
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
    print(f"average: {class_accuracies.mean().item() * 100:2.2f}")
    return

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


if __name__ == "__main__":
    validate_model(parser.parse_args())
