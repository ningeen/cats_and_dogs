import os
import random
import time
import logging
import logging.config

import yaml
import librosa
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .mobilenetv3 import mobilenetv3_small

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_PATH = 'pretrained/mobilenetv3-small-55df8e1f.pth'
TRAINED_PATH = 'pretrained/small_mobilenet_weights_3c.pt'
NUM_EPOCHS = 10
LR = 1e-3
SAMPLING_RATE = 16000
SAMPLING_DURATION = 2
SAVE_MODEL = True
INPUT_LENGTH = SAMPLING_RATE * SAMPLING_DURATION
LOADER_PARAMS = {'batch_size': 16, 'shuffle': True, 'num_workers': 4}

DOG_TRAIN_PATH = "input/train/dog/"
DOG_TEST_PATH = "input/test/test/"
CAT_TRAIN_PATH = "input/train/cat/"
CAT_TEST_PATH = "input/test/cats/"
OTHER_TRAIN_PATH = "input/16000/"
OTHER_TEST_PATH = "input/16000_test/"

APP_NAME = "training"
DEFAULT_LOGGING_CONFIG_FILE_PATH = "logging.conf.yml"
logger = logging.getLogger(APP_NAME)
with open(DEFAULT_LOGGING_CONFIG_FILE_PATH) as config_fin:
    logging.config.dictConfig(yaml.safe_load(config_fin))


def seed_everything(seed=1234):
    """Fix random seeds"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.debug("Seeds fixed.")


def read_file(path):
    """Read wav file"""
    wav, _ = librosa.core.load(path, sr=SAMPLING_RATE)
    wav, _ = librosa.effects.trim(wav)
    return wav


def read_wav_files(path):
    """Read all wav files from directory"""
    logger.debug("Reading files from %s", path)
    features = []
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            wav = read_file(os.path.join(path, filename))
            features.append(wav)
    logger.debug("Read %s files", len(features))
    return features


def get_mel(wav):
    """Calculate melspectrogram"""
    melspec = librosa.feature.melspectrogram(
        wav,
        sr=SAMPLING_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )
    logmel = librosa.core.power_to_db(melspec)
    return logmel


class CatDogDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, files, labels, transform=None, stable=False):
        """Init Dataset"""
        self.files = files
        self.labels = labels
        self.transform = transform
        self.stable = stable
        logger.debug("CatDogDataset initialized with %s files", len(self.files))

    def __len__(self):
        """Length"""
        return len(self.files)

    def __getitem__(self, index):
        """Generates one sample of data"""
        wav = self.files[index]
        label = self.labels[index]
        if len(wav) > INPUT_LENGTH:
            diff = len(wav) - INPUT_LENGTH
            if self.stable:
                start = 0
            else:
                start = np.random.randint(diff)
            end = start + INPUT_LENGTH
            wav = wav[start: end]
        else:
            diff = INPUT_LENGTH - len(wav)
            if self.stable:
                offset = 0
            else:
                offset = np.random.randint(diff)
            offset_right = diff - offset
            wav = np.pad(wav, (offset, offset_right), "constant")

        if self.transform:
            wav = self.transform(wav)

        mel = get_mel(wav)
        mel = mel[np.newaxis, :, :]
        return mel, label


def get_loader(dog_path, cat_path, other_path, params, transform=None):
    """Parse files in directory and create loader"""
    logger.debug("Creating loader for %s and %s", dog_path, cat_path)
    dog_data = read_wav_files(dog_path)
    cat_data = read_wav_files(cat_path)
    other_data = read_wav_files(other_path)
    data = dog_data + cat_data + other_data

    labels = np.zeros((len(data), 3), dtype=np.float32)
    labels[:len(dog_data), 0] = 1
    labels[len(dog_data): len(dog_data) + len(cat_data), 1] = 1
    labels[len(dog_data) + len(cat_data):, 2] = 1

    dataset = CatDogDataset(data, labels, transform=transform)
    dataloader = DataLoader(dataset, **params)
    return dataloader


def get_model(pretrained_mn3_path="", pretrained_path=""):
    """Load MobilenetV3 model with specified in and out channels"""
    logger.info("Loading MobilenetV3 model")
    model = mobilenetv3_small().to(DEVICE)
    if pretrained_mn3_path and not pretrained_path:
        model.load_state_dict(torch.load(pretrained_mn3_path))
        logger.debug("Weights loaded from %s", pretrained_mn3_path)

    model.features[0][0].weight.data = torch.sum(
        model.features[0][0].weight.data, dim=1, keepdim=True
    )
    model.features[0][0].in_channels = 1

    model.classifier[-1].weight.data = model.classifier[-1].weight.data[:3]
    model.classifier[-1].bias.data = model.classifier[-1].bias.data[:3]
    model.classifier[-1].out_features = 3

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
        logger.debug("Weights loaded from %s", pretrained_path)
    return model


def process_epoch(model, criterion, optimizer, loader):
    """Calc one epoch"""
    losses = []
    y_true = []
    y_pred = []
    with torch.set_grad_enabled(model.training):
        for local_batch, local_labels in loader:
            local_batch, local_labels = \
                local_batch.to(DEVICE), local_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(local_batch)
            probability = torch.softmax(outputs.data, dim=1)

            loss = criterion(outputs, torch.argmax(local_labels, dim=1))
            if model.training:
                loss.backward()
                optimizer.step()

            losses.append(loss)
            y_true.append(local_labels.detach().cpu().numpy())
            y_pred.append(probability.detach().cpu().numpy())
    loss_train = np.array(losses).astype(np.float32).mean()
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    auc_train = roc_auc_score(y_true, y_pred, multi_class='ovo')
    return loss_train, auc_train, y_true, y_pred


def train_model(model, criterion, optimizer, train_loader, test_loader):
    """Training loop"""
    logger.info("Start training model")
    logs = {'loss_train': [], 'loss_val': [], 'auc_train': [], 'auc_val': [], 'acc_val': []}
    best_true = None
    best_pred = None
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        # Training
        model.train()
        loss_train, auc_train, _, _ = \
            process_epoch(model, criterion, optimizer, train_loader)
        logs['loss_train'].append(loss_train)
        logs['auc_train'].append(auc_train)

        # Validation
        model.eval()
        loss_val, auc_val, y_true, y_pred = \
            process_epoch(model, criterion, optimizer, test_loader)
        acc_val = balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        logs['loss_val'].append(loss_val)
        logs['auc_val'].append(auc_val)
        logs['acc_val'].append(acc_val)
        logger.info(
            f"Epoch #{epoch + 1}. "
            f"Time: {(time.time() - start_time):.1f}s. "
            f"Train loss: {loss_train:.3f}, train auc: {auc_train:.3f}. "
            f"Val loss: {loss_val:.3f}, val auc: {auc_val:.3f}. "
            f"Acc: {acc_val:.3f}"
        )
        if acc_val >= np.max(logs['acc_val']):
            if SAVE_MODEL:
                torch.save(model.state_dict(), TRAINED_PATH)
            best_true = y_true
            best_pred = y_pred
    return best_true, best_pred


def main():
    """Train and save model"""
    logger.info("Start training script")
    seed_everything()

    train_loader = get_loader(DOG_TRAIN_PATH, CAT_TRAIN_PATH,
                              OTHER_TRAIN_PATH, LOADER_PARAMS)
    test_loader = get_loader(DOG_TEST_PATH, CAT_TEST_PATH,
                             OTHER_TEST_PATH, LOADER_PARAMS)

    model = get_model(PRETRAINED_PATH)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_true, best_pred = \
        train_model(model, criterion, optimizer, train_loader, test_loader)
    logger.info("Training done.")
