import logging
import logging.config

import yaml
from flask import Flask, render_template, request

from classifier import Classifier

AUDIO_PATH = "static/audio.wav"
app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)
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
        logger.debug('Audio file uploaded successfully')
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_page():
    """Page with model prediction"""
    if 'clf' not in locals():
        clf = Classifier()
    msg, label = clf.get_result_message(AUDIO_PATH)
    logger.debug('Prediction message: %s', msg)
    img = f"static/{label}.jpg"
    return render_template(
        "prediction.html",
        prediction_message=msg,
        prediction_image=img,
    )


if __name__ == "__main__":
    clf = Classifier()
    app.run(debug=True)
