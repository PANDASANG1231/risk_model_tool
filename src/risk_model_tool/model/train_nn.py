import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import *
from torchvision.transforms import *
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from transformers import get_linear_schedule_with_warmup


def fit(
    model,
    train_dataset,
    val_dataset,
    train_sampler_num=None,
    val_sampler_num=None,
    epochs=50,
    batch_size=32,
    warmup_prop=0.1,
    lr=1e-3,
    verbose=1,
    first_epoch_eval=0,
    device="cuda",
    loss_weight=3.5,
    model_dir=None,
):
    """
    Fitting function for the classification task.

    Args:
        model (torch model):
            Model to train.
        train_dataset (torch dataset):
             Dataset to train with.
        val_dataset (torch dataset):
            Dataset to validate with.
        epochs (int, optional):
            Number of epochs. Defaults to 50.
        batch_size (int, optional):
            Training batch size. Defaults to 32.
        lr (float, optional):
            Learning rate. Defaults to 1e-3.
        verbose (int, optional):
            Period (in epochs) to display logs at. Defaults to 1.
        first_epoch_eval (int, optional):
            Epoch to start evaluating at. Defaults to 0.
        device (str, optional):
            Device for torch. Defaults to "cuda".
        model_dir (str, optional):
            Dir for models. Defaults to None.


    Returns:
        numpy array [len(val_dataset)]: Last predictions on the validation data.
    """

    ###Check dict
    check_dict = {}

    NUM_WORKERS = 4
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # loss_fct = nn.BCEWithLogitsLoss()
    loss_fct = nn.CrossEntropyLoss(
        weight=torch.tensor([1, loss_weight], dtype=torch.float).to(device)
    )

    if model_dir is None:
        model_dir = "./model_folder"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if train_sampler_num is None:
        train_sampler_num = len(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        sampler=RandomSampler(train_dataset, num_samples=train_sampler_num),
    )

    if val_sampler_num is None:
        sampler = SequentialSampler(val_dataset)
    else:
        sampler = RandomSampler(val_dataset, num_samples=val_sampler_num)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        sampler=sampler,
    )

    num_training_steps = int(epochs * len(train_loader))
    num_warmup_steps = int(warmup_prop * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):

        print(f"Epoch {epoch} training")
        model.train()
        start_time = time.time()
        optimizer.zero_grad()
        avg_loss = 0

        for batch in tqdm(train_loader, total=len(train_loader)):

            images = batch[0].to(device).float()
            y_batch = batch[1].to(device)
            y_pred = model(images)

            loss = loss_fct(y_pred, y_batch)
            loss.backward()

            avg_loss += loss.item() / len(train_loader)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # for param in model.parameters():
            #     param.grad = None

        print(f"Epoch {epoch} avg loss: {avg_loss}")

        preds = np.empty(0)
        labels = np.empty(0)
        confidences = np.empty(0)

        model.eval()

        avg_val_loss, auc, precision, recall = 0.0, 0.0, 0.0, 0.0
        if epoch + 1 >= first_epoch_eval or epoch + 1 == epochs:
            print(f"Epoch {epoch} evaluating")
            with torch.no_grad():
                for batch in tqdm(val_loader, total=len(val_loader)):

                    images = batch[0].to(device).float()
                    y_batch = batch[1].to(device)

                    y_pred = model(images)

                    loss = loss_fct(y_pred, y_batch)
                    avg_val_loss += loss.item() / len(val_loader)

                    y_confidences = (
                        nn.Softmax(dim=1)(torch.sigmoid(y_pred))
                        .detach()
                        .cpu()
                        .numpy()[:, -1]
                    )
                    y_pred = torch.sigmoid(y_pred).argmax(axis=1).detach().cpu().numpy()

                    preds = np.concatenate([preds, y_pred])
                    labels = np.concatenate([labels, y_batch.detach().cpu().numpy()])
                    confidences = np.concatenate([confidences, y_confidences])

        try:
            auc = roc_auc_score(labels, confidences)
            precision = precision_score(labels, preds)
            recall = recall_score(labels, preds)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        except Exception as e:
            print("Could not calculate scores", str(e))

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s \t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )

            if epoch + 1 >= first_epoch_eval:
                print(
                    f"val_loss={avg_val_loss:.3f} \t auc={auc:.3f}\t precision={precision} \t recall={recall}"
                )
                print("Confusion metrics: (tn, fp, fn, tp) = ", (tn, fp, fn, tp))

            else:
                print("")

        check_dict["confidences"] = confidences
        check_dict["preds"] = preds
        check_dict["labels"] = labels
        check_dict["model"] = model

        torch.save(
            model,
            os.path.join(
                model_dir,
                f"checkpoint_epoch{epoch + 1:02d}_auc_{auc:.3f}_precision_{precision}_recall_{recall}.pkl",
            ),
        )

    return check_dict
