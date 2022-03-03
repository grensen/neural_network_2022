System.Action<string> print = System.Console.WriteLine;

// loads MNIST to this folder
print("Get oldSqlNet\n");

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

// ff
static void NetworkTest(AutoData d, int[] net, float[] weight)
{
    DateTime elapsed = DateTime.Now;
    int correct = 0, all = 0;
    int neuronLen = GetNeuronsLen(net);

    for (int x = 1; x < 10000 + 1; x++)
    {        
        // get target label
        int target = d.GetLabel(x - 1, false);
        // feed input sample
        float[] neuron = FeedSample(d.GetSample(x - 1, false), neuronLen);
        // prediction sample
        int prediction = FF(net, weight, neuron);
        // count predictions
        correct += prediction == target ? 1 : 0; all++; // true count
    } // runs end

    System.Console.WriteLine("Testing accuracy = " + (correct * 100.0 / all).ToString("F2") 
        + "%\nTime = " + (((TimeSpan)(DateTime.Now - elapsed)).TotalMilliseconds / 1000.0).ToString("F2") + "s\n");
}
// ff + bp
static void NetworkTraining(AutoData d, int[] net, float[] weight, int len, float lr, float mom)
{
    DateTime elapsed = DateTime.Now;
    int correct = 0, all = 0;
    int neuronLen = GetNeuronsLen(net);
    int layer = net.Length - 1;
    int output = net[layer]; // output neurons
    // correction value for each weight
    float[] delta = new float[weight.Length];
    int batch = 1;

    for (int x = 1; x < len + 1; x++)
    {        
        // get target label
        int target = d.GetLabel(x - 1, true);
        // feed input sample
        float[] neuron = FeedSample(d.GetSample(x - 1, true), neuronLen);
        // get prediction
        int prediction = FF(net, weight, neuron);
        // count predictions
        correct += prediction == target ? 1 : 0; all++; // true count
        // softmax with max trick
        Softmax(neuron, neuronLen, output);
        // backprop
        if (neuron[neuronLen - output + target] >= 0.99) continue;
        BP(net, weight, neuron, delta, target);
        batch++;
        // update
        if (prediction == target) continue;
        Update(net, weight, delta, layer, (neuronLen / layer * 1.0f) / (batch + 1), lr, mom);
        batch = 0;
    } // runs end

    System.Console.WriteLine("Training accuracy = " + (correct * 100.0 / all).ToString("F2")
        + "%\nTime = " + (((TimeSpan)(DateTime.Now - elapsed)).TotalMilliseconds / 1000.0).ToString("F2") + "s\n");
}
// input sample
static float[] FeedSample(Sample s, int neuronLen)
{
    float[] neuron = new float[neuronLen];
    for (int i = 0; i < 784; i++) neuron[i] = s.sample[i];
    return neuron;
}
// ff
static int FF(int[] net, float[] weight, float[] neuron)
{
    int layer = net.Length - 1, j = 0;
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
    // get prediction
    int prediction = 0, output = net[layer];
    float max = neuron[j];
    for (int i = 1; i < output; i++)
    {
        float n = neuron[i + j];
        if (n > max) { max = n; prediction = i; } // grab maxout prediction here
    }
    return prediction;
}
// softmax
static void Softmax(float[] neuron, int neuronLen, int output)
{
    float scale = 0;
    // softmax with max trick
    for (int n = neuronLen - output; n != neuronLen; n++) scale += neuron[n] = (float)MathF.Exp(neuron[n]);
    for (int n = neuronLen - output; n != neuronLen; n++) neuron[n] = neuron[n] / scale;
}
// bp
static void BP(int[] net, float[] weight, float[] neuron, float[] delta, int target)
{
    int neuronLen = neuron.Length;
    int weightLen = weight.Length;

    int layer = net.Length - 1;
    int input = net[0];
    int output = net[layer]; // output neurons
    int hidden = neuronLen - (input + output); // hidden neurons
    int inputHidden = neuronLen - output; // size of input and hidden neurons

    Span<float> gradient = new float[neuronLen - input];

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
}
// update
static void Update(int[] net, float[] weight, float[] delta, int layer, float mlt, float lr, float mom)
{
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
}
// glorot init     
static void Glorot(int[] net, float[] weight)
{
    Erratic rnd = new(12345);
    for (int i = 0, w = 0; i < net.Length - 1; i++, w += net[i - 0] * net[i - 1]) // layer
    {
        float sd = (float)Math.Sqrt(6.0f / (net[i] + net[i + 1]));
        for (int m = w; m < w + net[i] * net[i + 1]; m++) // weights
            weight[m] = rnd.NextFloat(-sd * 1.0f, sd * 1.0f);
    }
}
// neurons length
static int GetNeuronsLen(int[] net)
{
    int sum = 0;
    for (int n = 0; n < net.Length; n++) sum += net[n];
    return sum;
}
// weights length
static int GetWeightsLen(int[] net)
{
    int sum = 0;
    for (int n = 0; n < net.Length - 1; n++) sum += net[n] * net[n + 1];
    return sum;
}
// save network
static void NetworkSave(int[] net, float[] weight, string fullPath)
{
    int weightLen = weight.Length + 1;
    string[] netString = new string[weightLen];
    netString[0] = string.Join(",", net); // neural network at first line
    for (int i = 1; i < weightLen; i++)
        netString[i] = ((decimal)((double)weight[i - 1])).ToString(); // for precision
    File.WriteAllLines(fullPath, netString);
}
// load network
static void NetworkLoad(ref int[] net, ref float[] weight, string fullPath)
{
    FileStream Readfiles = new(fullPath, FileMode.Open, FileAccess.Read);
    string[] backup = File.ReadLines(fullPath).ToArray();

    // load network
    net = backup[0].Split(',').Select(int.Parse).ToArray();
    weight = new float[backup.Length - 1];
    // load weights
    for (int n = 1; n < backup.Length; n++)
        weight[n - 1] = float.Parse(backup[n]);
    Readfiles.Close(); // don't forget to close!
}

struct Sample
{
    public float[] sample;
    public int label;
}
struct AutoData // https://github.com/grensen/easy_regression#autodata
{
    public string source;
    public byte[] samplesTest, labelsTest;
    public byte[] samplesTraining, labelsTraining;

    public AutoData(string yourPath)
    {
        this.source = yourPath;

        // hardcoded urls from my github
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testnLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        // change easy names 
        string d1 = @"trainData", d2 = @"trainLabel", d3 = @"testData", d4 = @"testLabel";

        if (!File.Exists(yourPath + d1)
            || !File.Exists(yourPath + d2)
              || !File.Exists(yourPath + d3)
                || !File.Exists(yourPath + d4))
        {
            System.Console.WriteLine("\nData does not exist");
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

            // padding bits: data = 16, labels = 8
            System.Console.WriteLine("Download MNIST dataset from GitHub");
            this.samplesTraining = (new System.Net.WebClient().DownloadData(trainDataUrl)).Skip(16).Take(60000 * 784).ToArray();
            this.labelsTraining = (new System.Net.WebClient().DownloadData(trainLabelUrl)).Skip(8).Take(60000).ToArray();
            this.samplesTest = (new System.Net.WebClient().DownloadData(testDataUrl)).Skip(16).Take(10000 * 784).ToArray();
            this.labelsTest = (new System.Net.WebClient().DownloadData(testnLabelUrl)).Skip(8).Take(10000).ToArray();

            System.Console.WriteLine("Save cleaned MNIST data into folder " + yourPath + "\n");
            File.WriteAllBytes(yourPath + d1, this.samplesTraining);
            File.WriteAllBytes(yourPath + d2, this.labelsTraining);
            File.WriteAllBytes(yourPath + d3, this.samplesTest);
            File.WriteAllBytes(yourPath + d4, this.labelsTest); return;
        }
        // data on the system, just load from yourPath
        System.Console.WriteLine("Load MNIST data and labels from " + yourPath + "\n");
        this.samplesTraining = File.ReadAllBytes(yourPath + d1).Take(60000 * 784).ToArray();
        this.labelsTraining = File.ReadAllBytes(yourPath + d2).Take(60000).ToArray();
        this.samplesTest = File.ReadAllBytes(yourPath + d3).Take(10000 * 784).ToArray();
        this.labelsTest = File.ReadAllBytes(yourPath + d4).Take(10000).ToArray();
    }

    public Sample GetSample(int id, bool isTrain)
    {
        Sample s = new();
        s.sample = new float[784];

        if (isTrain) for (int i = 0; i < 784; i++)
                s.sample[i] = samplesTraining[id * 784 + i] / 255f;
        else for (int i = 0; i < 784; i++)
                s.sample[i] = samplesTest[id * 784 + i] / 255f;

        s.label = isTrain ? labelsTraining[id] : labelsTest[id];
        return s;
    }
    public float[] GetSampleF(int id, bool isTrain)
    {
        float[] sample = new float[784];
        if (isTrain) for (int i = 0; i < 784; i++)
                sample[i] = samplesTraining[id * 784 + i] / 255f;
        else for (int i = 0; i < 784; i++)
                sample[i] = samplesTest[id * 784 + i] / 255f;
        return sample;
    }
    public int GetLabel(int id, bool isTrain)
    {
        return isTrain ? labelsTraining[id] : labelsTest[id];
    }
    public void SaveWeights(string savePath, float[] weights)
    {
        Console.WriteLine("\nSave weights to " + source + @"myInfinityTest.txt");
        // bring weights into string
        string[] wStr = new string[weights.Length];
        for (int i = 0; i < weights.Length; i++)
            wStr[i] = ((decimal)((double)weights[i])).ToString(); // for precision
        // save weights to file
        File.WriteAllLines(source + savePath, wStr);
    }

    public float[] LoadWeights(string loadPath)
    {
        Console.WriteLine("\nLoad weights from " + source + @"myInfinityTest.txt");
        // load weights from file
        string[] wStr = File.ReadAllLines(source + loadPath);
        // string to float
        float[] weights = new float[wStr.Length];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = float.Parse(wStr[i]);
        return weights;
    }
}
class Erratic // https://jamesmccaffrey.wordpress.com/2019/05/20/a-pseudo-pseudo-random-number-generator/
{
    private float seed;
    public Erratic(float seed2)
    {
        this.seed = this.seed + 0.5f + seed2;  // avoid 0
    }
    public float Next()
    {
        double x = Math.Sin(this.seed) * 1000;
        double result = x - Math.Floor(x);  // [0.0,1.0)
        this.seed = (float)result;  // for next call
        return (float)result;
    }
    public float NextFloat(float lo, float hi)
    {
        float x = this.Next();
        return (hi - lo) * x + lo;
    }
};