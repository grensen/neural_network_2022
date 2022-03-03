# neural_network_2022
My latest highly efficient neural network implementation with IO support on a low level using C#

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

## FF

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

## BP

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
