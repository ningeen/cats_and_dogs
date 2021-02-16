import logging
import time

import numpy as np
import torch

from .train_model import read_file, get_model, CatDogDataset
from .train_model import DEVICE, config

APP_NAME = "classifier"
logger = logging.getLogger(APP_NAME)
TRAINED_PATH = config['trained_path']
INPUT_LENGTH = config['input_length']
LOADER_PARAMS = config['loader_params']


class Classifier:
    """Load model and predict class"""
    def __init__(self):
        """Initialize classifier"""
        self.device = DEVICE
        self.model = self.load_model()
        self.labels = {0: 'dog', 1: 'cat', 2: 'something_else'}

    def load_model(self):
        """Create model and load weights"""
        model = get_model(pretrained_path=TRAINED_PATH)
        model.eval()
        logger.debug("Model loaded, device: %s.", self.device)
        return model

    @staticmethod
    def get_batch(wav):
        """Get batch data"""
        chunks = list()
        weights = list()
        length = 0
        for i in range(LOADER_PARAMS['batch_size']):
            chunk = wav[length: length + INPUT_LENGTH]
            weight = len(chunk) / INPUT_LENGTH
            if weight > 0.25 or i == 0:
                weights.append(weight)
                chunks.append(chunk)
                length += INPUT_LENGTH
            else:
                break
        labels = np.array([None] * len(chunks), dtype=np.float32)
        logger.debug("Got batch with length: %s.", len(chunks))
        return chunks, labels, weights

    @staticmethod
    def get_spec(chunks, labels):
        """Get batch mel spectrogram"""
        dataset = CatDogDataset(chunks, labels, stable=True)
        data = list()
        for spec, _ in dataset:
            data.append(spec)
        data = np.array(data)
        logger.debug("Spec shape: %s.", data.shape)
        return data

    def get_data(self, wav):
        """Get preprocessed Tensor"""
        logger.debug("Audio shape: %s.", wav.shape)
        chunks, labels, weights = self.get_batch(wav)
        data = self.get_spec(chunks, labels)
        data = torch.tensor(data, dtype=torch.float32).to(DEVICE)
        logger.debug("Data shape: %s.", data.shape)
        return data, weights

    def predict(self, audio_path):
        """Predict class for wav-file saved in audio_path"""
        start = time.time()
        wav = read_file(audio_path)
        logger.debug("File read in %.3f s", time.time() - start)

        start = time.time()
        data, weights = self.get_data(wav)
        logger.debug("Got spec data in %.3f s", time.time() - start)

        start = time.time()
        with torch.no_grad():
            out = self.model(data)
        logger.debug("Prediction done in %.3f s", time.time() - start)

        logger.debug("Prediction shape: %s.", out.shape)
        proba = torch.softmax(out.data, dim=1).detach().cpu().numpy()
        proba_mean = np.average(proba, axis=0, weights=weights)
        return proba_mean

    def get_result_message(self, audio_path):
        """Return prediction message and class"""
        try:
            proba = self.predict(audio_path)
            label = np.argmax(proba)
            logger.info("Class probabilities: %s", proba)

            predicted_class = self.labels[label]
            predicted_proba = proba[label]
            msg = f"I'm {predicted_proba:.1%} sure it is a {predicted_class}."
            return msg.replace('_', ' '), predicted_class
        except Exception as error:
            logger.error("Prediction error: %s", error)
            return "Got some error, please try again.", "unknown"
