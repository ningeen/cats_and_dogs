import os
import random
import time
import logging
import logging.config

import yaml
import librosa
import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

from cats_and_dogs.mobilenetv3 import mobilenetv3_small

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_PATH = 'pretrained/mobilenetv3-small-55df8e1f.pth'
TRAINED_PATH = 'pretrained/small_mobilenet_weights.pt'
NUM_EPOCHS = 5
SAMPLING_RATE = 16000
SAMPLING_DURATION = 3
SAVE_MODEL = True
INPUT_LENGTH = SAMPLING_RATE * SAMPLING_DURATION
LOADER_PARAMS = {'batch_size': 16, 'shuffle': True, 'num_workers': 4}

DOG_TRAIN_PATH = "../cats_dogs/train/dog/"
DOG_TEST_PATH = "../cats_dogs/test/test/"
CAT_TRAIN_PATH = "../cats_dogs/train/cat/"
CAT_TEST_PATH = "../cats_dogs/test/cats/"

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
        label = self.labels[[index]]
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


def get_loader(dog_path, cat_path, params, transform=None):
    """Parse files in directory and create loader"""
    logger.debug("Creating loader for %s and %s", dog_path, cat_path)
    dog_data = read_wav_files(dog_path)
    cat_data = read_wav_files(cat_path)
    data = dog_data + cat_data
    labels = [0] * len(dog_data) + [1] * len(cat_data)
    labels = np.array(labels, dtype=np.float32)
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

    model.classifier[-1].weight.data = torch.sum(
        model.classifier[-1].weight.data, dim=0, keepdim=True
    )

    model.classifier[-1].bias.data = torch.sum(
        model.classifier[-1].bias.data, dim=0, keepdim=True
    )
    model.classifier[-1].out_features = 1

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
            probability = torch.sigmoid(outputs.data)

            loss = criterion(outputs, local_labels)
            if model.training:
                loss.backward()
                optimizer.step()

            losses.append(loss)
            y_true.append(local_labels.detach().cpu().numpy())
            y_pred.append(probability.detach().cpu().numpy())
    loss_train = np.array(losses).astype(np.float32).mean()
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    auc_train = roc_auc_score(y_true, y_pred)
    return loss_train, auc_train, y_true, y_pred


def train_model(model, criterion, optimizer, train_loader, test_loader):
    """Training loop"""
    logger.info("Start training model")
    logs = {'loss_train': [], 'loss_val': [], 'auc_train': [], 'auc_val': []}
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
        logs['loss_val'].append(loss_val)
        logs['auc_val'].append(auc_val)
        logger.info(
            f"Epoch #{epoch + 1}. "
            f"Time: {(time.time() - start_time):.1f}s. "
            f"Train loss: {loss_train:.3f}, train auc: {auc_train:.5f}. "
            f"Val loss: {loss_val:.3f}, val auc: {auc_val:.5f}"
        )
        if auc_val >= np.max(logs['auc_val']):
            if SAVE_MODEL:
                torch.save(model.state_dict(), TRAINED_PATH)
            best_true = y_true
            best_pred = y_pred
    return best_true, best_pred


def find_best_threshold(best_true, best_pred):
    """Calculate best threshold based on f1-score"""
    calc_f1 = lambda x: -f1_score(best_true, best_pred > x[0])
    res = minimize(calc_f1, np.array([0.5]), method='nelder-mead')
    best_thr = res.x[0]
    logger.info(f"Best thr: {best_thr}. F1-score: {-calc_f1(res.x):.5f}")


def main():
    """Train and save model"""
    logger.info("Start training script")
    seed_everything()

    train_loader = get_loader(DOG_TRAIN_PATH, CAT_TRAIN_PATH, LOADER_PARAMS)
    test_loader = get_loader(DOG_TEST_PATH, CAT_TEST_PATH, LOADER_PARAMS)

    model = get_model(PRETRAINED_PATH)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_true, best_pred = \
        train_model(model, criterion, optimizer, train_loader, test_loader)
    find_best_threshold(best_true, best_pred)
    logger.info("Training done.")


if __name__ == "__main__":
    main()
