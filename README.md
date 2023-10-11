### Objective
The main purpose of the project is to improve ML in Production skills, research how image classification models work.

### Data
The data we are using are taken from [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) Kaggle competition. More info can be found at the competition website. In a nutshell we want to label satellite image chips with atmospheric conditions and various classes of land cover/land use. It is a multi-labeling problem with 17 different classes. In the competition algorithms were scored using the mean F2 score.

Here we only use the jpg images. Note that zip file should be unzipped.The data can still be downloaded [here](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data)

### Pipeline preparation

1. Creating and activating the environment
    ```
    python3 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Installing packages

    In the activated environment:
    ```
    pip install -r requirements.txt
    ```

3. Customise [config.yaml](configs/config.yaml) to suit your needs.
Pay attention to `config.train_classes_path`, you need to specify where the dataset file `train_classes.csv` was downloaded to, 
and in `data_dir` you need to specify where the `train-jpg` folder was downloaded to.

### Training

Start training:

```
PYTHONPATH=. python src/train.py configs/config.yaml

```

### Load best model using dvc
Command

```
dvc pull weights/vgg16_feature_extractor.pth.dvc

```

will create `vgg16_feature_extractor.pth` within `weights` directory, it will be used as a base model for prediction

### Inference

command

```
PYTHONPATH=. python src/predict.py --image_path <path_to_image>

```

will be used as a inference point for <path_to_image> image, test_image is provided in the root of repo

### Experiment Logs

[Link to VGG16 experiment in ClearML](https://app.clear.ml/projects/02b3dd2c5ecb4c779d6b33f995b15890/experiments/d465c59eb5f14b9682c0a71c1f31332d/output/execution)
