
import torch
import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
from sklearn.utils.class_weight import compute_class_weight

from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures, get_f1_measure
from agent.classifiers.utils.early_stopping import EarlyStopping


def train_loop(model, learning_rate, epsilon, batch_size, epochs, device, train_ds, val_ds,
               for_task_label, binary_classifier, use_class_weight=False, use_early_stopping=True, patience=2):

    early_stopping = EarlyStopping(patience=patience)
    best_model = None

    train_dataloader = DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        batch_size=batch_size
    )

    if use_class_weight:
        train_labels = np.array([row[1].item() for row in train_ds])
        class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        if class_weights.size(dim=0) < model.num_labels:
            # class_weights = torch.cat([class_weights, torch.zeros(model.num_labels) - class_weights.size(dim=0)], dim=-1)
            class_weights = torch.cat([torch.zeros(model.num_labels - class_weights.size(dim=0)), class_weights], dim=-1)
        class_weights = class_weights.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

    starting_time = datetime.now()

    for epoch in range(epochs):
        model.train()
        for _, batch in enumerate(train_dataloader):

            batch_input_ids =   batch[0].to(device) # lstm: [batch size, seq_lenght]
            batch_labels =      batch[1].to(device) # lstm: [batch size]

            _, batch_pred = model(batch_input_ids)  # lstm: [batch size, num_classes]

            batch_loss = loss_fn(input=batch_pred, target=batch_labels)

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        class_predictions, labels = eval_loop(model, batch_size, device, val_ds)

        current_time = datetime.now()
        remaining_time = (epochs - epoch + 1) * (current_time - starting_time) / (epoch + 1)
        get_evaluation_measures(f"Epoch: {epoch + 1}/{epochs}, remaining time: {remaining_time}", labels, class_predictions,
                                for_task_label, binary_classifier)

        if use_early_stopping:
            f1 = get_f1_measure(labels, class_predictions)
            if early_stopping.stop_training(f1, model):
                best_model = early_stopping.get_best_model()
                break

    if best_model is not None:
        return best_model
    else:
        return model


def eval_loop(model, batch_size, device, dataset):

    validation_dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size
    )

    predictions = []
    labels = []
    for _, batch in enumerate(validation_dataloader):

        batch_input_ids =   batch[0].to(device)
        batch_labels =      batch[1].to(device) # en evaluación no tiene sentido pasar los labels al modelo

        with torch.no_grad():
            _, batch_pred = model(batch_input_ids)

        batch_pred =    batch_pred.detach().cpu()
        batch_labels =  batch_labels.detach().cpu()

        predictions.append(batch_pred)
        labels.append(batch_labels)

    flat_predictions =      torch.cat(predictions, dim=0)
    class_predictions =     torch.max(flat_predictions, dim=1)[1]
    labels =                torch.cat(labels, dim=0)

    return class_predictions, labels


def test_loop(model, batch_size, device, dataset):

    validation_dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size
    )

    predictions = []
    for _, batch in enumerate(validation_dataloader):

        batch_input_ids =   batch[0].to(device)

        with torch.no_grad():
            _, batch_pred = model(batch_input_ids)

        batch_pred =    batch_pred.detach().cpu()

        predictions.append(batch_pred)


    flat_predictions =      torch.cat(predictions, dim=0)
    class_predictions =     torch.max(flat_predictions, dim=1)[1]

    return class_predictions, flat_predictions
