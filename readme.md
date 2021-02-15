# Cats and dogs

Flask app which allows distinguish cats and dogs.

Model was trained using [Audio Cats and Dogs](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs) kaggle dataset.

## Running web application
Run flask app:
```
python wsgi.py
```

## Train model
Model based on MobilenetV3. Pytorch implementation got from [this repo](https://github.com/d-li14/mobilenetv3.pytorch).  
Model ovo roc_auc 0.968, balanced accuracy 0.874.  
To retrain model run: 
```
python train.py
```