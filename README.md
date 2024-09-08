# iFood 2019 challlenge
This repo contains the project of the unsupervised learning course.
The requirements were:

- Model below 1 million parameters
- Implement supervised learning task with CNN
- Implement self supervised lerarning task
- Use the self supervised learning pretrained model to extract features and train a traditional classifier with them
- [BONUS] fine tune the self supervised model on the supervised task (untested but implemented)

## How to run
It is assumed that cuda GPU is available in the system.
Install the requirements (conda environment reccomended):
```
pip insall -r requirements.txt
```
### Preprocessing
Download, clean and augment the data.
```
python data_preprocessing.py
```
This command has the following options:
```
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   directory to store the preprocessed data into
  --download-dir DOWNLOAD_DIR
                        directory to store the downloaded data into. The download size is bout 3 Gb
  --remove-src          remove the source tar files
  --train-size TRAIN_SIZE
                        size of the train set. Must be in [0,1]
  --generate-ssl        generate the self supervised learning datasets
  --ssl-perms SSL_PERMS
                        number of permutations of the generated self supervised learning datasets
  --clean-data          whether or not to clean the data
```
The result are three folder containing the training, validation and test splits.

### Training
Train the supervised model
```
python training.py
```
To train other types of models, see the following options:
```
  -h, --help            show this help message and exit
  --train-dir TRAIN_DIR
                        directory containing the training data
  --val-dir VAL_DIR     directory containing the training data
  --weights-dir WEIGHTS_DIR
                        directory to store the trained model weights into
  --type TYPE           type of training. Supported values are (full, sup, selfsup, classifier)
  --use-ssl-pretrained  whether or not to use self supervised pretrained weights.Affects both sup and full training types
  --ssl-permutations SSL_PERMUTATIONS
                        path to the file containing the jigsaw permutations
```

### Evaluation
To plot the evaluation metrics of the validation set run the following:
```
python testing.py
```
To cahange the data see the following options:
```
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   directory containing the evaluation data
  --classifier-dir CLASSIFIER_DIR
                        directory containing the traditional classifier
  --cnn-checkpoint CNN_CHECKPOINT
                        path to the Pytorch Lightning model checkpoint
```
