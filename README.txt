Name: Abhishek Pillai
UNI:  arp2247

COMS 4701 - Coding 3: Sign Language MNIST CNN

------------------------------------------------------------------
Answers to the questions
------------------------------------------------------------------

1. One-hot encoding turns an integer label into a vector with a 1 at that class's index and 0 elsewhere
   (e.g. label 3 -> [0,0,0,1,0,...]). It matters because categorical_crossentropy compares the predicted
   probability vector with a target vector, and it avoids implying that label 4 is "bigger" than label 3.
   In Keras: keras.utils.to_categorical(labels, num_classes=25).

2. Dropout randomly sets a fraction of a layer's units to zero at each training step (and is switched off
   at prediction time). This stops units from co-adapting and acts like averaging many thinned
   sub-networks, which reduces overfitting.

3. ReLU is max(0, x): it is cheap and does not saturate for positive inputs, so gradients keep flowing in
   deep networks. Sigmoid squashes values into (0, 1) and saturates at both ends, which gives vanishing
   gradients and slower training.

4. Softmax converts the 25 raw output scores into a probability distribution (non-negative, sums to 1),
   which is what categorical cross-entropy expects and lets us take the argmax as the predicted letter.

5. Convolution (100x100x1 input, 16 filters, 5x5 kernel, stride 1, valid padding): 100 - 5 + 1 = 96, so the
   output is 96 x 96 x 16. MaxPooling with pool_size (2,2): 96 / 2 = 48, so the output is 48 x 48 x 16.

6. In the final run the validation accuracy (1.0000) and the test accuracy (1.0) were both 100%, so there
   was no gap. In general, though, validation accuracy is the optimistic number. The images in train.csv
   look like augmented copies of a small set of original photos, so a random train/validation split puts
   near-duplicates of training images into the validation set. I measured this: each validation image is on
   average 0.52 away (pixel units 0-1) from its nearest training image and a 1-nearest-neighbour classifier
   gets 99.9% of them right, whereas test images are 2.45 away on average and 1-NN only gets 81%. The test
   set is therefore the harder, more honest measure, and early in training validation accuracy runs far
   ahead of what the model can really do on unseen images.

------------------------------------------------------------------
Model and workflow
------------------------------------------------------------------

Data: pixels scaled to [0, 1] and reshaped to 28x28x1; labels one-hot encoded with 25 classes (labels 0-24,
class 9 / J is never used); 90/10 stratified train/validation split (random_state=42). predict() applies the
same scaling to the raw 0-255 test rows and returns argmax class labels.

Architecture (Keras Sequential, ~441k parameters):
  - Augmentation layers (active only during training): RandomRotation(0.06), RandomZoom(0.1),
    RandomTranslation(0.1, 0.1). No flips, because hand signs are not mirror-symmetric. This helps because
    the test images are not near-copies of the training images.
  - Conv(32,3x3) -> BatchNorm -> Conv(32,3x3) -> MaxPool -> Dropout(0.25)
  - Conv(64,3x3) -> BatchNorm -> Conv(64,3x3) -> MaxPool -> Dropout(0.25)
  - Conv(128,3x3) -> BatchNorm -> MaxPool -> Dropout(0.25)
  - Flatten -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(25, softmax)
  All convolutions use ReLU and 'same' padding.

Training: Adam, categorical_crossentropy, batch size 128 (the default of train()), 30 epochs, with
ReduceLROnPlateau (monitor val_loss, factor 0.5, patience 3) to lower the learning rate when validation
loss stalls. Trained locally on CPU in about 7 minutes with TensorFlow 2.21 / Keras 3.15 on Python 3.12.

Result: test accuracy 1.0 (100%) on the provided test.csv with the grading cells run unmodified.
