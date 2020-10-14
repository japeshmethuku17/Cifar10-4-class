# Cifar10-4-class
Development of CNN model from scratch to classify images belonging to bird, cat, dog and horse.

The image dataset is contained in Cifar_Images which contains three folders test, train and validation and has 2000, 3000 and 15000 images respectively. There are 4 classes in total - bird, cat, dog, horse.

The architecture of the CNN model is shown by plotting the figure and has been placed in 'Model' folder of this project. The hdf5 file of the developed CNN model is also contained in the same folder.

The figures for learning curves of the model training is shown in the 'Results' folder. The model statistics - training accuracy, training loss, validation accuracy, validation loss, test accuracy and test loss is shown in the 'execution_result.xlsx' file.

I have included the python script and the IPython Notebook in the 'Scripts' folder. I used Google Colaboratory to design and implement the CNN model. The code has been documented well to enable a proper understanding of the implementation procedure.

The architecture of the model is as follows:

(Conv2D -> BN -> Conv2D -> BN -> MaxPool) -> (Conv2D -> BN -> Conv2D -> BN -> MaxPool) -> DO -> (Conv2D -> BN -> Conv2D -> BN -> MaxPool) -> DO -> Flatten -> Dense -> BN -> DO -> Softmax layer

Conv2D - Convolution layer
BN - Batch Normalization layer
MaxPool - Max Pooling layer
DO - Dropout layer
Dense - Fully Connected layer

'ModelCheckPoint' callback was used throughout the training process to save the best model after each step in the training process.

The CNN model was implemented using tensorflow 2 and keras deep learning framework.