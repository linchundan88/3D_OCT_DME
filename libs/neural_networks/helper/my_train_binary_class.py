''' train a binary classifier '''
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from torch.cuda.amp import autocast, GradScaler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#amp:AUTOMATIC MIXED PRECISION
def train(model, loader_train, criterion, activation, optimizer, scheduler, epochs_num, label_smoothing=0.,
          amp=False, accumulate_grads_times=None,
          log_interval_train=10, log_interval_valid=None,
          loader_valid=None, loader_test=None,
          save_model_dir=None, save_jit=False):

    if torch.cuda.device_count() > 0:
        model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if amp:
        scaler = GradScaler()

    for epoch in range(epochs_num):
        print(f'Epoch {epoch}/{epochs_num - 1}')
        model.train()
        epoch_loss, epoch_corrects = 0, 0
        running_loss, running_sample_num, running_corrects = 0, 0, 0
        list_labels, list_preds = [], []

        for batch_idx, (inputs, labels) in enumerate(loader_train):
            list_labels.extend(labels.numpy())

            labels = labels.float()
            if label_smoothing != 0:
                labels[labels == 1] = 1-label_smoothing
                labels[labels == 0] = label_smoothing
            inputs = inputs.to(device)
            labels = labels.to(device)
            with autocast(enabled=amp):
                outputs = model(inputs)
                outputs = torch.flatten(outputs)
                loss = criterion(outputs, labels)

            if (accumulate_grads_times is None) or  \
                    (accumulate_grads_times is not None and batch_idx % accumulate_grads_times == 0):
                optimizer.zero_grad()
                if not amp:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if not amp:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)

            # region batch statistics
            epoch_loss += loss.item()
            running_loss += loss.item()  #reduction='mean'

            assert activation in [None, 'sigmoid'], f'activation function error!'
            if activation == 'sigmoid':
                outputs = torch.sigmoid(outputs)
            outputs = outputs.cpu().detach().numpy()
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            list_preds.extend(outputs)

            labels = labels.cpu().numpy()
            if label_smoothing > 0:
                labels[labels > 0.5] = 1
                labels[labels <= 0.5] = 0

            running_corrects += np.sum(outputs == labels)
            running_sample_num += inputs.shape[0]
            if log_interval_train is not None:
                if batch_idx % log_interval_train == log_interval_train - 1:
                    print(f'[epoch:{epoch}, batch:{batch_idx}] losses:{running_loss / log_interval_train:8.3f}, acc:{running_corrects / running_sample_num:8.2f}')
                    running_loss, running_corrects, running_sample_num = 0, 0, 0
            #endregion

        scheduler.step()

        print(f'epoch{epoch} losses:{epoch_loss / (batch_idx+1):8.3f}')
        print('Confusion Matrix of training dataset:', confusion_matrix(list_labels, list_preds))
        # print(classification_report(list_labels, list_preds))

        if loader_valid:
            print('computing validation dataset...')
            loss_valid = validate(model, loader_valid, log_interval_valid, criterion=criterion, activation=activation)
        if loader_test:
            print('computing test dataset...')
            validate(model, loader_test, log_interval_valid, activation=activation)

        if save_model_dir is not None:
            os.makedirs(save_model_dir, exist_ok=True)

            if save_jit:
                #torch.jit.script(model) error
                if loader_valid:
                    save_model_file = os.path.join(save_model_dir, f'epoch{epoch}_{round(loss_valid, 3)}.pth_jit')
                else:
                    save_model_file = os.path.join(save_model_dir, f'epoch{epoch}.pth_jit')
                print('save model:', save_model_file)
                scripted_module = torch.jit.script(model)
                torch.jit.save(scripted_module, save_model_file)
                #model = torch.jit.load(model_file_saved)
            else:
                if loader_valid:
                    save_model_file = os.path.join(save_model_dir, f'epoch{epoch}_{round(loss_valid, 3)}.pth')
                else:
                    save_model_file = os.path.join(save_model_dir, f'epoch{epoch}.pth')
                os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
                print('save model:', save_model_file)
                try:
                    state_dict = model.module.state_dict()
                except AttributeError:
                    state_dict = model.state_dict()
                torch.save(state_dict, save_model_file)


@torch.no_grad()
def validate(model, dataloader, log_interval=None, criterion=None, activation=None):
    model.eval()
    epoch_loss = 0
    list_labels, list_preds = [], []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        list_labels.extend(labels.numpy())
        labels = labels.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = torch.flatten(outputs)
        if criterion is not None:
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
        assert activation in [None, 'sigmoid'], f'activation function error!'
        if activation == 'sigmoid':
            outputs = torch.sigmoid(outputs)

        outputs = outputs.cpu().detach().numpy()
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        list_preds.extend(outputs)

        if log_interval is not None:
            if batch_idx % log_interval == log_interval - 1:
                print(f'batch:{batch_idx + 1} ')

    if criterion is not None:
        print(f'losses:{epoch_loss / (batch_idx+1):8.3f}')
    print('Confusion Matrix:', confusion_matrix(list_labels, list_preds))
    # print(classification_report(list_labels, list_preds))

    return epoch_loss / (batch_idx+1)

