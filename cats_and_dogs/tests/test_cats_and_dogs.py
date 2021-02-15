import logging
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cats_and_dogs import classifier
from cats_and_dogs import run_flask
from cats_and_dogs import train_model as tm


MEL_LENGTH = 126
SAMPLE_SHAPE = (1, 1, 128, MEL_LENGTH)
SAMPLES_DIR = "static"
CAT_SAMPLE_PATH = "static/cat_sample.wav"
DOG_SAMPLE_PATH = "static/dog_sample.wav"


def test_classifier_load_model():
    clf = classifier.Classifier()
    out = clf.predict(CAT_SAMPLE_PATH)
    assert isinstance(clf.model, torch.nn.Module), "No pytorch model"
    assert 3 == out.shape[0], f"Wrong output shape: {out.shape}"


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
    spec, _ = clf.get_data(spec)
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
    yield run_flask.app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_page(app, client):
    res = client.get('/')
    page_text = res.data.decode()

    assert res.status_code == 200
    assert 'Cats and Dogs' in page_text
    assert 'Submit' in page_text


@patch('cats_and_dogs.run_flask.AUDIO_PATH', DOG_SAMPLE_PATH)
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
    return tm.get_loader(SAMPLES_DIR, SAMPLES_DIR, SAMPLES_DIR, tm.LOADER_PARAMS)


def test_loader(loader):
    for local_batch, local_labels in loader:
        local_batch = local_batch.detach().cpu().numpy()
        local_labels = local_labels.detach().cpu().numpy()
        break
    assert local_batch.shape[0] >= 4
    assert local_batch.shape[1:] == torch.Size(SAMPLE_SHAPE)[1:], \
        "Wrong sample shape"
    assert all(local_labels.max(axis=1) == 1) and \
           all(local_labels.min(axis=1) == 0), \
        "Bad labels"


def test_process_epoch(loader):
    model = tm.get_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    loss_train, auc_train, y_true, y_pred = \
        tm.process_epoch(model, criterion, optimizer, loader)
    assert np.isfinite(loss_train), "Nan/inf Loss error"
    assert np.isfinite(auc_train), "Nan/inf AUC error"
    assert y_true.shape[0] != 0, "y_true zero shape"
    assert y_pred.shape[0] != 0, "y_pred zero shape"
    assert np.all(np.isfinite(y_pred)), "Nan/inf in prediction"


@patch('cats_and_dogs.train_model.SAVE_MODEL', False)
@patch('cats_and_dogs.train_model.NUM_EPOCHS', 1)
def test_train_model(loader):
    model = tm.get_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    best_true, best_pred = tm.train_model(
        model, criterion, optimizer, loader, loader
    )
    assert np.all(np.isfinite(best_pred)), \
        "Nan/inf in prediction"
