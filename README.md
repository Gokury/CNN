# CNN
Evolution of CNN algorithm 



ANN means neural networks in general. So a CNN is a ANN. CNNs are not only used for Image Processing,
they can also be applied to Text like Text Comprehension.
Down to how much the image size will be reduced for the feature detector?
In the practical section it will be reduced down to 64 by 64 dimensions.


We are building feature maps with each convolutional filter, meaning we are making features that help us
classify the object. The example shown in the Intuition Lecture is like a one-dimensional edge detection
filter. That is one feature map. So we want our model to "activate" in a feature map only where there is
an edge. We will have several feature maps like this which when all put together will help us identify the
object. This is helped by removing the black or the negative values.
What is the purpose of the ReLU?

The biggest reason why we use ReLU is because we want to increase the non-linearity in our image. And
ReLU acts as a function which breaks up linearity. And the reason why we want to break up linearity in our
network is because images themselves are highly non-linear. Indeed they have a lot of non-linear elements
like the transitions between pixels.

Why does Max-Pooling consider only 4 values to take a maximum from and not 2 or 8 values?
Also in convolution how is the feature detector formed?
Because a maximum is taken from a spot on the image which is represented by a 2x2 square on our image.
Therefore it covers 4 pixels.
After flattening, we going to get one long vector for all pooled layers or a vector for each
pooled layer


The forward propagation of the input images happening inside the CNN by describing in a few words:

Sequential is first used to specify we introduce a sequence of layers, as opposed to a computational graph.
Convolution2D is then used to add the convolutional layer.
MaxPooling2D is then used to apply Max Pooling to the input images.
Flatten is then used to flatten the pooled images.
Dense is used to add the output layer with softmax.

