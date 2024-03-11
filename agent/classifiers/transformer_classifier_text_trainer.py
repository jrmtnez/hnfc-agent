import torch
import numpy as np

from datetime import datetime
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight

from agent.classifiers.evaluation.evaluation_mgmt import get_evaluation_measures, get_f1_measure
from agent.classifiers.utils.early_stopping import EarlyStopping


def train_loop(accelerator, model, learning_rate, epsilon, warmup_steps, batch_size, epochs, device, train_ds, val_ds,
               for_task_label, binary_classifier, use_class_weight=False, use_early_stopping=True, patience=2):

    early_stopping = EarlyStopping(patience=patience)
    best_model = None

    train_dataloader = DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        batch_size=batch_size
    )

    if use_class_weight:
        train_labels = np.array([row[2].item() for row in train_ds])
        class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        if class_weights.size(dim=0) < model.num_labels:
            class_weights = torch.cat([torch.zeros(model.num_labels - class_weights.size(dim=0)), class_weights], dim=-1)

        class_weights = class_weights.to(device)

    no_of_train_batches = len(train_dataloader)
    total_train_steps = no_of_train_batches * epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

    starting_time = datetime.now()

    for epoch in range(epochs):
        model.train()
        for _, batch in enumerate(train_dataloader):

            # batch_input_ids =           batch[0].to(device)
            # batch_attention_masks =     batch[1].to(device)
            # batch_labels =              batch[2].to(device)
            batch_input_ids =           batch[0]
            batch_attention_masks =     batch[1]
            batch_labels =              batch[2]

            if use_class_weight:
                output = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels, class_weights=class_weights)
            else:
                output = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)

            batch_loss = output[0]

            optimizer.zero_grad()
            # batch_loss.backward()
            accelerator.backward(batch_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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

        batch_input_ids =           batch[0].to(device)
        batch_attention_masks =     batch[1].to(device)
        batch_labels =              batch[2]

        with torch.no_grad():
            output = model(batch_input_ids, attention_mask=batch_attention_masks)

        batch_pred =    output[0]
        batch_pred =    batch_pred.detach().cpu()

        predictions.append(batch_pred)
        labels.append(batch_labels)

    flat_predictions =      torch.cat(predictions, dim=0)
    class_predictions =     torch.max(flat_predictions, dim=1)[1]
    labels =                torch.cat(labels, dim=0)

    return class_predictions, labels


def test_loop(model, batch_size, device, dataset):

    test_dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size
    )

    predictions = []
    for _, batch in enumerate(test_dataloader):

        batch_input_ids =           batch[0].to(device)
        batch_attention_masks =     batch[1].to(device)

        with torch.no_grad():
            output = model(batch_input_ids, attention_mask=batch_attention_masks)

        batch_pred =    output[0]
        batch_pred =    batch_pred.detach().cpu()

        predictions.append(batch_pred)

    flat_predictions =      torch.cat(predictions, dim=0)
    class_predictions =     torch.max(flat_predictions, dim=1)[1]

    return class_predictions, flat_predictions
