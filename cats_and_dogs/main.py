import logging
import logging.config
import time

import yaml
import soundfile as sf
from flask import Flask, render_template, request

from cats_and_dogs.classifier import Classifier
from cats_and_dogs.train_model import read_file, SAMPLING_RATE

AUDIO_PATH = "static/audio.wav"
app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)
app.clf = Classifier()
APP_NAME = "cat_dogs_demo"
DEFAULT_LOGGING_CONFIG_FILE_PATH = "logging.conf.yml"
logger = logging.getLogger(APP_NAME)
with open(DEFAULT_LOGGING_CONFIG_FILE_PATH) as config_fin:
    logging.config.dictConfig(yaml.safe_load(config_fin))


@app.route("/", methods=["POST", "GET"])
def index_page():
    """Main page, saves audio to server"""
    if request.method == "POST":
        audio_data = request.files['audio_data']
        with open(AUDIO_PATH, 'wb') as audio:
            audio_data.save(audio)

        wav = read_file(AUDIO_PATH)
        sf.write(AUDIO_PATH, wav, SAMPLING_RATE)

        logger.debug('Audio file uploaded successfully')
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_page():
    """Page with model prediction"""
    start = time.time()
    msg, label = app.clf.get_result_message(AUDIO_PATH)
    logger.debug(
        'Prediction done in %.1f s message: %s',
        time.time() - start, msg
    )
    img = f"static/{label}.jpg"
    return render_template(
        "prediction.html",
        prediction_message=msg,
        prediction_image=img,
    )
