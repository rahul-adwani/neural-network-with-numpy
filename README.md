# Building  a Neural Network Using Only Numpy

### In this exercise, I am going to demonstrate the working of Neural Network and how it can be coded using only Numpy

## What is a Neural Network?
### A Neural network is a collection of neurons which transmit, receive, store, share and process information. The human brain handles information in the form of a neural network. An artificial neural network on the other hand, tries to mimic the human brain function and is one of the most important areas of study in the domain of Artificial Intelligence and Machine Learning.

## How does a Neural Network work?
#### A human brain in the form of a neural network has gazillions of neurons and each neuron receive, process and transmit information to and from other neurons. Each neuron helps the human being to understand one part of the information and in combination, all the neurons give us a complete picture.
#### Below is the picture of one biological neuron. Here, the dendrites receive the input signals and pass on to the cell body where it gets processed and accumulated. The output signal(s) are passed on to the next neuron's dendrites as synapse through the axon.

![Image - Biological Neuron](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/neuron-296581_640.png)

#### However, in the case of an artificial neuron, the network comprises of multiple layers and each layer has multiple neurons. In principle, the neuron in an artificial neural network does the same job of receiving, processing, storing and transmitting information, however, in practice they are a simplified version of the biological neuron. Below is the picture of an artificial neuron. Here, the input signals are the data inputs, importance of the information is defined by the weight, each neuron is expected to have a bias and the outputs are the result of the dot product of the information weights and the inputs to the neuron.

![Image - Artificial Neuron](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/artificial%20neuron%20self.png)

## Building a Neural Network
#### Now that we have understood the basics of a neural network, let us try and build one. Below is the architecture that we are going to use for this exercise:

![Image - Artificial Neuron](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/BP_L5.jpg)

#### Here we have the following entities:

 1. Data - Here we are going to use the famous IRIS dataset. Here is the [link](https://raw.githubusercontent.com/rahul-adwani/neural-network-with-numpy/main/data/irisTestData.csv) to the dataset
 2. Network - This network is a fully connected network where each neuron is connected to the other. This network has 6 layers including the input and output layers.
 3. Layers - The first layer is the input layer. In the current architecture, we have 6 inputs. Let us call them x1, x2, x3, x4, x5 and x6. The next 4 layers are the hidden layers with 6, 6, 4 and 5 neurons each. The last layer is the output layer which has only one neuron. The output of this neuron is going to be the probability of the output variable 'y' being 1 or 0.
 4. Weights - Each neuron has a set of weights associated to it. The nomenclature of each weight is defined by the input neuron # in its layer, the current neuron # in its layer and the layer #
 5. Biases - Each neuron has a bias associated to it. For simplicity, I have assumed bias = 0 for all the neurons
 6. The nomenclature for inputs to a neuron is A<sub>y</sub><sup>x</sup>. Here x is the layer # and y is the neuron # in its layer
 7. The nomenclature for the weights of a neuron is W<sub>ab</sub><sup>x</sup>. Here x is the layer #, a/b are the neuron # in their respective layers.
 8. The nomenclature for the activation function is &sigma;<sub>y</sub><sup>x</sup>. Here y is the layer # and x is the neuron # in its layer. Having mentioned that, I have defined activation functions at the level of a layer and not at the level of a neuron.
 
 ## Initialising a Neural Network
 
 #### Now that we have built the network and that we are familiar with the entities present in the network, let us try to understand the operations that happen inside a network:
1. Random Initialisation - The first step after defining the architecture of a neural network is to initialise the weights and biases associated with the network's neurons. There are a lot of methods available over the internet, I have chosen to do a random initialisation using Numpy:

    weights = np.random.randn(n_inputs) * 0.1, bias = np.random.randn(1)*0.1
 
## Forward Propagation

- Once we have initialised the neurons of the network, the second step if forward propagation. Forward Propagation is the step where the neurons actually process and transmit information to each other. In an artificial neuron, the mathematical operation is to multiply the weights and the inputs and add bias to the product:

![Image - Input to Activation Function](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/input%20to%20activation.JPG)

- This value is then passed to the activation function. The reason for using an activation function in a neural network is to introduce a non-linearity. For example, I have used tanh function as one of the activation functions in the network. Below is its equation:

![Image - Tanh Function](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/tanh.JPG)

- This activated output of a neuron is transmitted to the next neuron as an input and this process is followed until the last layer's neurons. The last layer gives us the output of the network.

## Calculating Loss

- Once the input data has traversed through the entire network, the last layer of the network gives us the predicted output. In order to ascertain, how close is the output with the expected one, we calculate loss. There are a lot of methods using which loss can be calculated, but since I have built this network for binary classification, I have used binary cross entropy. Below is the formula for calculating the binary cross entropy loss:

![Image - Cost Function](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/cost%20function.JPG)

## Backward Propagation

- Now that we have calculated loss for one round of traversal of inputs from input layer to the output layer, the output is fed back to the network from the output to the input layer. This process is called back propagation. In this process, the weights and biases are updated by using different optimisation methods. The one that I have used is Stochastic Gradient Descent. In this method, we use one input sample at a time to do a complete traversal from input to output and back to output.
- Back propagation is the most tricky part of the network operations. This process gives us a way to train/optimise the weights and biases (or the parameters of the network) which empowers the network to make the predictions as accurate as possible.
- In order to understand this in depth, we have to be brushed up with core Mathematical concepts like Linear Algebra, Partial Differentiation, Matrix Multiplication, Chain Rule, etc. You can skip the below part in case you are not interested in knowing the Mathematics going on behind the scenes.
- The way to update the weights using back propagation is by using the below relationship:

![Image - Gradient Descent](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/Grad%20Descent.JPG)

Here, &alpha; is the learning rate. Using this we can control the rate at which the weights get trained and updated.
In order to perform the partial differentiation, we need to understand the **Chain Rule**. In order to demonstrate the Chain Rule, I will try to differentiate the loss with the first weight in each layer:

Layer #5 (Output Layer)
![Image - Back Propagation Layer #5](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/BP_L5_1.jpg)
In the diagram above, the highlighted part shows the flow of information, from last layer to the Layer # 5.
We know that, back propagation helps us in optimise/update weights and biases by using the below relation:

![Image - Gradient Descent](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/Grad%20Descent.JPG)

Now, we already have the weights with us, we generally decide &alpha; using hyperparameter tuning. The only part left to calculate is the partial differentiation term.
For this calculation, we shall use the concept of Chain rule. Let us try to differentiate the loss function with respect to the weights associated with this layer, one by one:

![Image - Layer 5 Equations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/layer%205.JPG)

Layer 4 (Hidden Layer)
![Image - Back Propagation Layer #4](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/BP_L4_1.jpg)
In this diagram, we can see the flow of information from the output layer to Layer #4. We can express this information flow in the following equation for back propagation:

![Image - Layer 4 Equations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/layer%204.JPG)

Layer 3 (Hidden Layer)
![Image - Back Propagation Layer #3](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/BP_L3_1.jpg)
In this diagram, we can see the flow of information from the output layer to Layer #3. We can express this information flow in the following equation for back propagation:

![Image - Layer 3 Equations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/layer%203_1.JPG)

When simplified, this equation looks like this:

![Image - Layer 3 Equations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/layer%203_2.JPG)

This equation might look very daunting but if you try to write it down in your notebook, you can see a pattern. We have summation signs in this equation due to the fact that from the third layer onwards, the output of the neurons traverse into all the neurons in the succeeding layers, hence during the back propagation as well, the flow would be via all the neurons in the outer layers.

![Image - Layer 3 Equations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/layer%203_3.JPG)

Layer 2 (Hidden Layer)
![Image - Back Propagation Layer #2](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/BP_L2_1.jpg)
In this diagram, we can see the flow of information from the output layer to Layer #2. Here, I have shown only two flows, however, the flow would happen from all the neurons of Layer #4, since all the neurons in this layer are affected by the weight.
We can express this information flow in the following equation for back propagation:

![Image - Layer 2 Equations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/layer%202_1.JPG)

When simplified, this equation looks like this:

![Image - Layer 2 Equations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/layer%202_2.JPG)

If this equation looks scary, then just try to copy it on your notebook and see that there is a definite pattern. Similar to the previous layer equation, we have summation signs in this equation due to the fact that from the third layer onwards, the output of the neurons traverse into all the neurons in the succeeding layers, hence during the back propagation as well, the flow would be via all the neurons in the outer layers.

![Image - Layer 2 Equations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/layer%202_3.JPG)

Layer 1 (Hidden Layer)
![Image - Back Propagation Layer #1](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/BP_L1_1.jpg)
In this diagram, we can see the flow of information from the output layer to Layer #1. Here, I have shown only three flows, however, the flow would happen from all the neurons of Layer #4 & Layer #3, since all the neurons in these layer are affected by the weight.
We shall not go into the equation for this layer, even though it will have an easy-to-find pattern but it will be a huge equation. Rather, let us jump directly to the matrix of derivatives:

![Image - Layer 1 Equations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/layer%201_1.JPG)

### We observed here that all we have to do is to calculate &delta;<sub>y</sub><sup>x</sup> and multiply it by the inputs to the neuron and would have the derivative.

#### Applying a little more of Linear Algebra in the above equations, we have the below simplified relations:


![Image - Final Delta Calculations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/final%20delta%20calc%201.JPG)

![Image - Final Delta Calculations](https://raw.github.com/rahul-adwani/neural-network-with-numpy/main/images/final%20delta%20calc%202.JPG)

#### We can see that all we have to calculate is &delta;<sub>1</sub><sup>5</sup> and rest all the &delta;s can be calculated one by one.

## Training
- Now that we have done all the three steps of Forward Propagation, Loss Calculation and Back Propagation, we can go ahead and train the data with multiple epochs. For this exercise, I have trained the model with a batch size of 1 across all the samples.

## Evaluation
- In order to evaluate a binary classification model, we can build a confusion matrix and use different metrics like accuracy, precision, recall - depending upon the representation of classes and business requirements.

## Further Exercises
- Further to this, we can use different and more efficient set of optimisers, callbacks, dropout, etc

## Some points to note:
- In order to calculate the derivatives, I have assumed &delta;<sub>y</sub><sup>x</sup> * W<sub>ab</sub><sup>x</sup> as the backward output of Neuron #y of Layer #x. In similarity to Forward Propagation where the output of a neuron is fed to an activation function, in Back Propagation, the backward output is fed to a differentiation function which is denoted as f&prime;(Z<sub>y</sub><sup>x</sup>)
- If we observe the patterns in the equations above, we can see that the best approach to solve these equations would be to use the power of Vectorisation of the Numpy library.
- I have used Python for building this neural network with Pandas and Numpy libraries. Also, the entire exercise is done keeping OOPS concept in the back of mind.

I have shared the notebook in this repository. You can check and share your comments with me.

You can check out my other repositories here: [Github](https://github.com/rahul-adwani?tab=repositories)
You can contact me here: [LinkedIn](https://www.linkedin.com/in/rahuladwani/)
If you like the content, please give it a star

Thanks for reading,
Rahul Adwani
