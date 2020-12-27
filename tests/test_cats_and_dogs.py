import logging
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, './cats_and_dogs/')
import classifier
import main
import train_model as tm


MEL_LENGTH = 188
SAMPLE_SHAPE = (1, 1, 128, 188)
SAMPLES_DIR = "static"
CAT_SAMPLE_PATH = "static/cat_sample.wav"
DOG_SAMPLE_PATH = "static/dog_sample.wav"


def test_classifier_load_model():
    clf = classifier.Classifier()
    out = clf.predict(CAT_SAMPLE_PATH)
    assert isinstance(clf.model, torch.nn.Module), "No pytorch model"
    assert np.isfinite(out), f"Output is not a number: {out.item()}"


@pytest.fixture()
def clf():
    return classifier.Classifier()


@pytest.mark.parametrize(
    "spec_length",
    [
        pytest.param(1, id='one_lengths'),
        pytest.param(tm.INPUT_LENGTH - 1, id='small_lengths'),
        pytest.param(tm.INPUT_LENGTH + 1, id='more_lengths'),
        pytest.param(tm.INPUT_LENGTH * 100, id='big_lengths'),
        pytest.param(tm.INPUT_LENGTH, id='equal_lengths'),
    ]
)
def test_process_spectrogram(spec_length, clf):
    spec = np.random.normal(size=(SAMPLE_SHAPE[2]))
    spec = clf.get_data(spec)
    assert spec.shape[-1] == MEL_LENGTH, \
        "Wrong spectrogram length after preprocessing"


@pytest.mark.parametrize(
    "sample_path, label",
    [
        pytest.param(CAT_SAMPLE_PATH, "cat", id='cat'),
        pytest.param(DOG_SAMPLE_PATH, "dog", id='dog'),
    ]
)
def test_model_work_correct_on_samples(sample_path, label, clf, caplog):
    msg, predicted_class = clf.get_result_message(sample_path)
    assert all(record.levelno <= logging.WARNING for record in
               caplog.records), "ERROR captured in log"
    assert label == predicted_class


@pytest.fixture
def app():
    yield main.app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_page(app, client):
    res = client.get('/')
    page_text = res.data.decode()

    assert res.status_code == 200
    assert 'Cats and Dogs' in page_text
    assert 'Submit' in page_text


@patch('main.AUDIO_PATH', DOG_SAMPLE_PATH)
def test_prediction_page(app, client, caplog):
    res = client.post('/prediction')
    page_text = res.data.decode()
    assert res.status_code == 200
    assert '% sure it is a dog' in page_text
    assert 'Return' in page_text
    assert all(record.levelno <= logging.WARNING for record in
               caplog.records), "ERROR captured in log"


@pytest.fixture
def loader():
    return tm.get_loader(SAMPLES_DIR, SAMPLES_DIR, tm.LOADER_PARAMS)


def test_loader(loader):
    for local_batch, local_labels in loader:
        break
    assert local_batch.shape[0] >= 4
    assert local_batch.shape[1:] == torch.Size(SAMPLE_SHAPE)[1:], \
        "Wrong sample shape"
    assert all([label in [0, 1] for label in local_labels]), \
        "Bad labels"


def test_process_epoch(loader):
    model = tm.get_model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    loss_train, auc_train, y_true, y_pred = \
        tm.process_epoch(model, criterion, optimizer, loader)
    assert np.isfinite(loss_train), "Nan/inf Loss error"
    assert np.isfinite(auc_train), "Nan/inf AUC error"
    assert y_true.shape[0] != 0, "y_true zero shape"
    assert y_pred.shape[0] != 0, "y_pred zero shape"
    assert np.all(np.isfinite(y_pred)), "Nan/inf in prediction"


@patch('train_model.SAVE_MODEL', False)
@patch('train_model.NUM_EPOCHS', 1)
def test_train_model(loader):
    model = tm.get_model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    best_true, best_pred = tm.train_model(
        model, criterion, optimizer, loader, loader
    )
    tm.find_best_threshold(best_true, best_pred)
    assert np.all(np.isfinite(best_pred)), \
        "Nan/inf in prediction"
