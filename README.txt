Name: Abhishek Pillai
UNI:  arp2247

COMS 4701 - Coding 3: Sign Language MNIST

=========================================================
Answers to the questions
=========================================================

1. What is one-hot encoding, and why does it matter?
   It turns a label like 3 into a vector with a 1 in position 3 and 0s everywhere else. The network's
   output layer gives one probability per class, and categorical cross-entropy needs a target in the same
   shape. It also stops the model from treating label 4 as "more" than label 3. In Keras I used
   to_categorical(labels, num_classes=25).

2. What is dropout, and how does it help with overfitting?
   Dropout randomly switches off some of a layer's units at each training step, so the network can't lean
   on any one unit and has to spread what it learns around. It is only active during training. (My final
   model doesn't use it; see the notes below.)

3. How does ReLU differ from sigmoid?
   ReLU is just max(0, x), so it is cheap and its gradient doesn't shrink for positive inputs. Sigmoid
   squashes everything into (0, 1) and flattens out at both ends, which makes gradients vanish and slows
   training in deeper networks.

4. Why do we need softmax in the output layer?
   It turns the 25 raw scores into probabilities that are positive and add up to 1. That is what
   cross-entropy expects, and it lets us read off the most likely letter with argmax.

5. Output sizes for the example network:
   Conv layer (100x100x1 input, 16 filters, 5x5 kernel, stride 1, no padding): 100 - 5 + 1 = 96, so the
   output is 96 x 96 x 16. After 2x2 max pooling it is 48 x 48 x 16.

6. Final validation accuracy vs. test accuracy - which is higher, and why?
   Validation accuracy is much higher: it reached 100% (1.0000) while test accuracy was only about 91%.
   The images in train.csv look like many augmented copies of a smaller set of original photos, so when I
   split off a validation set at random, almost every validation image has a near-twin in the training set.
   I checked this: the average distance (pixels scaled to 0-1) from a validation image to its closest
   training image is 0.52, and a simple nearest-neighbour classifier gets 99.9% of them right. For test
   images the average distance is 2.45 and the same classifier only gets 81%. So the validation score
   mostly measures memory, and the test score is the honest one.

=========================================================
About the model
=========================================================

Preparing the data: I scale pixels from 0-255 down to 0-1, reshape each row into a 28x28x1 image, and
one-hot encode the labels with 25 classes (labels run 0-24; 9 / J never appears, so that output unit just
goes unused). I hold out a random 10% of train.csv as the validation set. predict() applies the same
scaling to the raw test rows and returns the argmax class as an integer.

The network is a small, plain CNN (about 227k parameters):

   Conv2D(32, 3x3, ReLU) -> MaxPooling(2x2)
   Conv2D(64, 3x3, ReLU) -> MaxPooling(2x2)
   Flatten -> Dense(128, ReLU) -> Dense(25, softmax)

I compile it with Adam and categorical cross-entropy, and train for 30 epochs at the default batch size of
128 in train(). Everything runs locally on CPU (TensorFlow/Keras, Python 3.12) and takes a few minutes.

Results: about 91% test accuracy. I trained it three separate times and got 91.4%, 91.1% and 91.0%, which
clears the 90% mark but not by a lot. Training and validation accuracy both hit about 100% within a few
epochs, so those curves say very little about how it does on new images.

How I got here: I started from a bigger model with random rotation/zoom/shift augmentation, batch
normalization, dropout and a learning-rate scheduler, and that scored close to 100% on the test set. I
then deliberately stripped it back to the simple version above, because nothing in the assignment requires
those extras and I wanted a model I could fully explain. The cost was roughly nine points of test
accuracy, which suggests the extras (most likely the augmentation) were doing real work. I did not test
them one at a time, so I can't say which one mattered most.
