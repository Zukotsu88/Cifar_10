# Cifar_10
This is a convolutional neural-network based image classifier that runs off of [a Keras backend](https://keras.io/) in order to be trained across the [Cifar_10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). It uses a sequential model beginning with a pretrained [VGG16 model](https://keras.io/applications/#vgg16) and appending four more hidden layers. 

In this dataset, there are a total of 10 classes with 6000 images per class. Additionally, there are a total of five training batches and
one test batch, each with 1000 images each. 

There also exists a similar [Keras example](https://keras.io/examples/cifar10_cnn/) on their website.

![cifar_10](https://i2.wp.com/appliedmachinelearning.blog/wp-content/uploads/2018/03/cifar2.jpg?fit=427%2C325&ssl=1)
