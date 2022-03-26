# neural_network_2022

![alt text](https://raw.githubusercontent.com/grensen/gif_test/master/Figures/gg_one_hello_goodgame.gif?raw=true)

Neuronal networks bring perception into context. In the example above, a handwritten number is converted into a classification of what number it is.

The animation shows my work [from 2020](https://github.com/grensen/gif_test), and serves as a reference for the more modern implementation of neural networks in 2022 presented in the demo.

## Demo

<p align="center">
  <img src="https://github.com/grensen/neural_network_2022/blob/main/figures/demo.png?raw=true">
</p>

The main steps as follows. First, the network is initialized and then the MNIST training data set is trained, followed by the test. The network is then saved and reloaded to verify that the test result is the same as it appears to be.

## High Level Code

~~~cs
// declare net, learning rate and momentum
int[] u       = { 784, 16, 16, 10 };
var lr        = 0.005f;
var momentum  = 0.5f;

// get workspace on the machine
float[] weight = new float[GetWeightsLen(u)];
// get random weights to transmit signals with breaking symmetry
Glorot(u, weight);
// get MNIST data
AutoData d = new(@"C:\mnist\");
// MNIST training 60k
NetworkTraining(d, u, weight, 60000, lr, momentum);
// MNIST test 10k 
NetworkTest(d, u, weight);
// save trained network
NetworkSave(u, weight, (d.source + "myNN.txt"));

// reload trained network and test again
int[] loadedNet = { }; 
float[] loadedWeight = { };
NetworkLoad(ref loadedNet, ref loadedWeight, (d.source + "myNN.txt"));
NetworkTest(d, loadedNet, loadedWeight);
print("End demo");
~~~

To run the demo you need visual studio 2022 with dotnet 6 or higher. The Autodata class then does the work of providing the input and output capabilities.

At a high level, this is the code.

## Modularized

~~~cs
// functions
static void NetworkTest(AutoData d, int[] net, float[] weight) {}...
static void NetworkTraining(AutoData d, int[] net, float[] weight, int len, float lr, float mom) {}...
static float[] FeedSample(Sample s, int neuronLen){}...
static int FF(int[] net, float[] weight, float[] neuron) {}...
static void Softmax(float[] neuron, int neuronLen, int output) {}...
static void BP(int[] net, float[] weight, float[] neuron, float[] delta, int target) {}...
static void Update(int[] net, float[] weight, float[] delta, int layer, float mlt, float lr, float mom) {}...  
static void Glorot(int[] net, float[] weight) {}...
static int GetNeuronsLen(int[] net) {}...
static int GetWeightsLen(int[] net) {}...
static void NetworkSave(int[] net, float[] weight, string fullPath) {}...
static void NetworkLoad(ref int[] net, ref float[] weight, string fullPath) {}...
// helper
struct Sample {}...
struct AutoData {}...
class Erratic {};...
~~~

It may seem unusual, but this project is still connected to an upcoming project and so my decision was made for a complete modularization of the individual steps. The code could also be designed differently and maybe it should be.

## FF

<p align="center">
  <img src="https://github.com/grensen/neural_network_2022/blob/main/figures/ff.gif?raw=true">
</p>

---

~~~cs
for (int i = 0, k = net[0], w = 0; i < layer; i++)
{
    int left = net[i], right = net[i + 1];
    for (int l = 0; l < left; l++)
    {
        float n = neuron[j + l];
        if (n > 0) for (int r = 0; r < right; r++)
                neuron[k + r] += n * weight[w + r];
        w += right;
    }
    j += left; k += right;
}
~~~

The forward pass, we go across layer i, from input neurons j to output neurons k. 
First the input neuron on the left side is checked, which corresponds to the relu activation. If the value of the left neuron is more than 0, the neuron is added to all output neurons. This means that the code calculates input length times output length. Which creates the well known fully connected pattern whereby each neuron on the input side is individually connected to each neuron on the output side. Important here is that cache locallity is given.

## BP

<p align="center">
  <img src="https://github.com/grensen/neural_network_2022/blob/main/figures/bp.gif?raw=true">
</p>

---

~~~cs
// output error gradients, hard target as 1 for its class
for (int n = inputHidden, nc = 0; n < neuronLen; n++, nc++)
    gradient[n - input] = target == nc ? 1 - neuron[n] : -neuron[n];

for (int i = layer - 1, j = hidden, ww = weightLen, k; i >= 0; i--)
{
    int left = net[i], right = net[i + 1];
    ww -= right * left; k = j; j -= left;

    // hidden gradients - from last hidden to first hidden
    if (i != 0)
        for (int l = 0, w = ww; l < left; l++)
        {
            float gra = 0, n = neuron[input + j + l];
            if (n > 0)
                for (int r = 0; r < right; r++)
                    gra += weight[w + r] * gradient[k + r];
            w += right;
            gradient[j + l] = gra;
        }
    // all deltas
    for (int l = 0, w = ww; l < left; l++)
    {
        float n = neuron[input + j + l];
        if (n > 0)
            for (int r = 0; r < right; r++)
                delta[w + r] += n * gradient[k + r];
        w += right;
    }
}
~~~

Instead of perceptron wise way of working, this code runs layer wise. As also the animation backwards shows. What is particularly interesting about this implementation is that the way back is based on the way forward. 

Which means that a clean layer wise implementation forward only needs 3 small changes to go back and reduce the error. Instead of going from the first to the last layer, we need to go backwards from the last to the first layer.
By calculating the gradient instead of the activation, we have also unpacked this step to finally calculate the delta value for each weight in the network.

Before I end this very dirty explanation, I would like to briefly remind you of a fundamental idea. Neural networks seem super complex, but you can make them simple if you understand this:
~~~
Forwards:
neuronOutputRight += neuronInputLeft * weight
Backwards:
gradientInputLeft += weight * gradientOutputRight
Update:
weight += neuronInputLeft * gradientOutputRight
~~~

## Update

~~~cs
for (int i = 0, mStep = 0; i < layer; i++, mStep += net[i - 0] * net[i - 1]) // layers
{
    float oneUp = (float)Math.Sqrt(2.0f / (net[i + 1] + net[i])) * mlt;
    for (int m = mStep, mEnd = mStep + net[i] * net[i + 1]; m < mEnd; m++) // weights 
    {
        float del = delta[m], s2 = del * del;
        if (s2 > oneUp) continue; // check overwhelming deltas
        weight[m] += del * lr;
        delta[m] = del * mom;
    }
}
~~~

Modern networks also use a delta step, so that training steps are used in batches rather than after each individual backproagation.

