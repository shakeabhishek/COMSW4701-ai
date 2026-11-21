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
   on any one unit and has to spread what it learns around. It is only active during training. My model
   uses it (0.25 after the pooling/conv layers, 0.5 before the output), and adding it on its own raised
   test accuracy by about four points.

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
   Validation accuracy is higher: it reached 100% (1.0000) while test accuracy was about 96-97%.
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

The network is a small CNN (about 243k parameters):

   Conv2D(32, 3x3, ReLU) -> MaxPooling(2x2) -> Dropout(0.25)
   Conv2D(64, 3x3, ReLU) -> MaxPooling(2x2) -> Dropout(0.25)
   Conv2D(128, 3x3, ReLU) -> Dropout(0.25)
   Flatten -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(25, softmax)

I compile it with Adam and categorical cross-entropy, and train for 30 epochs at the default batch size of
128 in train(). Everything runs locally on CPU (TensorFlow/Keras, Python 3.12) and takes about five
minutes per run.

Results: test accuracy of about 96-98%. I trained this exact model four times and got 96.5% (the run saved
in the notebook), 97.2%, 96.1% and 98.0%. Training and validation accuracy both hit about 100% within a
few epochs, so those curves say very little about how it does on new images.

How I got here: I first built a bigger model with rotation/zoom/shift augmentation, batch normalization,
dropout and a learning-rate scheduler, which scored close to 100% on the test set. I then stripped
everything back to a plain two-conv CNN to see what actually mattered. That plain version scored about
91% (91.4%, 91.1%, 91.0%). I then added pieces back one at a time, running each version three times:

   plain two-conv CNN               91.0 - 91.4%
   + a third conv layer             91.6 - 93.1%
   + dropout (two convs)            94.7 - 95.9%
   + dropout and third conv layer   96.1 - 98.0%   <- final model

Dropout gave the biggest single jump and the two changes together roughly add up. I did not test
augmentation, batch normalization or a learning-rate schedule separately in this simpler setup, so I can't
say how much they would add. A caveat: I compared these versions using the test set, since the validation
set can't tell them apart, so the final number is a little optimistic compared with fresh data. With only
three runs per version, differences under about half a point are probably just noise.
