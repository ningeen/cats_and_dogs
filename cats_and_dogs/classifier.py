import logging

import numpy as np
import torch

from cats_and_dogs.train_model import read_file, get_model, CatDogDataset
from cats_and_dogs.train_model import DEVICE, TRAINED_PATH

APP_NAME = "classifier"
logger = logging.getLogger(APP_NAME)


class Classifier:
    """Load model and predict class"""
    def __init__(self):
        """Initialize classifier"""
        self.device = DEVICE
        self.model = self.load_model()
        self.labels = {0: 'dog', 1: 'cat'}

    def load_model(self):
        """Create model and load weights"""
        model = get_model(pretrained_path=TRAINED_PATH)
        model.eval()
        logger.debug("Model loaded, device: %s.", self.device)
        return model

    @staticmethod
    def get_data(wav):
        """Get preprocessed Tensor"""
        logger.debug("Audio shape: %s.", wav.shape)
        labels = np.array([None], dtype=np.float32)
        dataset = CatDogDataset([wav], labels, stable=True)
        data, _ = dataset[0]
        logger.debug("Spec shape: %s.", data.shape)
        data = data[np.newaxis, :, :, :]
        data = torch.tensor(data, dtype=torch.float32).to(DEVICE)
        logger.debug("Data shape: %s.", data.shape)
        return data

    def predict(self, audio_path):
        """Predict class for wav-file saved in audio_path"""
        wav = read_file(audio_path)
        data = self.get_data(wav)
        out = self.model(data)
        proba = torch.sigmoid(out.data).item()
        return proba

    def get_result_message(self, audio_path):
        """Return prediction message and class"""
        try:
            proba = self.predict(audio_path)
            label = int(np.round(proba))
            logger.info("Class probability: %5f.", proba)

            predicted_class = self.labels[label]
            predicted_proba = proba if label else 1 - proba
            msg = f"I'm {predicted_proba:.1%} sure it is a {predicted_class}."
            return msg, predicted_class
        except Exception as error:
            logger.error("Prediction error: %s", error)
            return "Got some error, please try again.", "unknown"
