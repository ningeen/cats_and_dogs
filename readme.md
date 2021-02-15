# Cats and dogs

Flask app which allows distinguish cats and dogs.

Model was trained on [Audio Cats and Dogs](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs) kaggle dataset.

## Running web application
Run flask app:
```
python wsgi.py
```

## Train model
Model based on MobilenetV3. Pytorch implementation got from https://github.com/d-li14/mobilenetv3.pytorch  
Model roc_auc 0.98, f1-score 0.975.  
To retrain model run: 
```
python train.py
```