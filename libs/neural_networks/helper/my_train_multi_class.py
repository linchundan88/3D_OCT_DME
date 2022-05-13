''' train a binary classifier '''
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#amp:AUTOMATIC MIXED PRECISION
def train(model, loader_train, criterion, activation, optimizer, scheduler, epochs_num,
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
            inputs = inputs.to(device)
            labels = labels.to(device)

            if (accumulate_grads_times is None) or  \
                    (accumulate_grads_times is not None and batch_idx % accumulate_grads_times == 0):
                optimizer.zero_grad()
            with autocast(enabled=amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            if (accumulate_grads_times is None) or  \
                    (accumulate_grads_times is not None and batch_idx % accumulate_grads_times == 0):
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
            assert activation in [None, 'softmax'], f'activation function error!'
            if activation == 'softmax':
                outputs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            list_labels.extend(labels.cpu().numpy().tolist())
            list_preds.extend(preds.cpu().numpy().tolist())
            running_corrects += torch.sum(preds == labels).cpu().numpy()
            running_sample_num += inputs.shape[0]

            if log_interval_train is not None:
                if batch_idx % log_interval_train == log_interval_train - 1:
                    print(f'[epoch:{epoch}, batch:{batch_idx}] losses:{running_loss / log_interval_train:8.3f}, acc:{running_corrects / running_sample_num:8.2f}')
                    running_loss, running_corrects, running_sample_num = 0, 0, 0
            #endregion

        scheduler.step()

        print(f'epoch{epoch} losses:{epoch_loss / (batch_idx+1):8.3f}')
        print('Confusion Matrix of training dataset:', confusion_matrix(list_labels, list_preds))
        print(classification_report(list_labels, list_preds))

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
                save_model_file = os.path.join(save_model_dir, f'epoch{epoch}{round(loss_valid, 3)}.pth_jit')
                print('save model:', save_model_file)
                scripted_module = torch.jit.script(model)
                torch.jit.save(scripted_module, save_model_file)
                #model = torch.jit.load(model_file_saved)
            else:
                save_model_file = os.path.join(save_model_dir, f'epoch{epoch}{round(loss_valid, 3)}.pth')
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
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        if criterion is not None:
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
        assert activation in [None, 'softmax'], f'activation function error!'
        if activation == 'softmax':
            outputs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        list_labels.extend(labels.cpu().numpy().tolist())
        list_preds.extend(preds.cpu().numpy().tolist())

        if log_interval is not None:
            if batch_idx % log_interval == log_interval - 1:
                print(f'batch:{batch_idx + 1} ')

    print('Confusion Matrix:', confusion_matrix(list_labels, list_preds))
    print(classification_report(list_labels, list_preds))

    if criterion is not None:
        print(f'losses:{epoch_loss / (batch_idx+1):8.3f}')
        return epoch_loss / (batch_idx+1)
