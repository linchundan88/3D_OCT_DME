
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix


def train(model, loader_train, criterion, optimizer, scheduler,
          epochs_num,
          log_interval_train=10, log_interval_valid=None, save_model_dir=None,
          loader_valid=None, loader_test=None, accumulate_grads_times=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(epochs_num):
        print(f'Epoch {epoch}/{epochs_num - 1}')
        model.train()
        epoch_loss, epoch_sample_num, epoch_corrects = 0, 0, 0
        running_loss, running_sample_num, running_corrects = 0, 0, 0

        list_labels, list_preds = [], []

        for batch_idx, (inputs, labels) in enumerate(loader_train):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if accumulate_grads_times is None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                if batch_idx % accumulate_grads_times == 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # statistics
            outputs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            list_labels += labels.cpu().numpy().tolist()
            list_preds += preds.cpu().numpy().tolist()

            #show average losses instead of batch total losses  *inputs.size(0) total losses
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.gradients).cpu().numpy()
            running_sample_num += len(inputs)

            epoch_loss += loss.item()
            epoch_corrects += torch.sum(preds == labels.gradients).cpu().numpy()
            epoch_sample_num += len(inputs)

            if log_interval_train is not None:
                if batch_idx % log_interval_train == log_interval_train - 1:
                    print(f'[epoch:{epoch}, batch:{batch_idx}] losses:{running_loss / log_interval_train:8.2f}, acc:{running_corrects / running_sample_num:8.2f}')
                    running_loss,running_corrects, running_sample_num = 0, 0, 0

        print(f'epoch{epoch} losses:{epoch_loss / (batch_idx+1):8.2f}, acc:{epoch_corrects / epoch_sample_num:8.2f}')
        print('Confusion Matrix of training dataset:', confusion_matrix(list_labels, list_preds))

        scheduler.step()

        if loader_valid:
            print('compute validation dataset...')
            validate(model, loader_valid, log_interval_valid)
        if loader_test:
            print('compute test dataset...')
            validate(model, loader_test, log_interval_valid)

        if save_model_dir:
            save_model_file = os.path.join(save_model_dir, f'epoch{epoch}.pth')
            os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
            print('save model:', save_model_file)
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            torch.save(state_dict, save_model_file)

# @torch.no_grad()
def validate(model, dataloader, log_interval=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    running_sample_num, running_corrects = 0, 0
    list_labels, list_preds = [], []

    if True:
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            list_labels += labels.cpu().numpy().tolist()
            list_preds += preds.cpu().numpy().tolist()

            running_corrects += torch.sum(preds == labels.gradients).cpu().numpy()
            running_sample_num += len(inputs)

            if log_interval is not None:
                if batch_idx % log_interval == log_interval - 1:
                    print(f'batch:{batch_idx + 1} acc:{ running_corrects / running_sample_num:8.2f}')
                    running_corrects, running_sample_num = 0, 0

    print('Confusion Matrix:', confusion_matrix(list_labels, list_preds))



