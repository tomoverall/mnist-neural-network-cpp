#include <iostream>  // Standard input/output
#include <vector>    // Vectors
#include <cmath>     // Math functions
#include <cstdlib>   // Random number generation
#include <ctime>     // Time functions
#include <fstream>   // File input/output
#include <string>    // Strings
#include <sstream>   // String streams
#include <iterator>  // Iterators
#include <algorithm> // Algorithms
#include <random>    // Random number generation

using namespace std;

// Sigmoid function
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x)
{
    return x * (1 - x);
}

// Neuron structure
struct Neuron
{
    double output;          // Output of the neuron
    double delta;           // Delta of the neuron for backpropagation
    vector<double> weights; // Weights connected neurons
};

// Layer structure
struct Layer
{
    vector<Neuron> neurons; // Neurons of the layer
};

// Network structure
class Network
{
private:
    vector<Layer> layers; // Layers of the network
    double eta;           // Learning rate

public:
    // Network constructor
    Network(const vector<int> &sizes, double learning_rate) : eta(learning_rate)
    {
        srand(time(NULL)); // Seed random number generator

        // Create layers
        for (int i = 0; i < sizes.size(); i++)
        {
            layers.push_back(Layer());
            for (int j = 0; j < sizes[i]; j++)
            {
                layers[i].neurons.push_back(Neuron());
                if (i > 0)
                {
                    for (int k = 0; k < sizes[i - 1]; k++)
                    {
                        layers[i].neurons[j].weights.push_back((double)rand() / RAND_MAX); // populate the objects we just made with random weights
                    }
                }
            }
        }
    }

    // Feed forward
    void feed_forward(const vector<double> &input)
    {
        // Set input values equal to the output of the input layer
        for (size_t i = 0; i < input.size(); i++)
        {
            layers[0].neurons[i].output = input[i]; // first layer is the input layer
        }

        // forward propogate
        for (size_t i = 1; i < layers.size(); i++)
        {
            for (size_t j = 0; j < layers[i].neurons.size(); j++)
            {
                double sum = 0.0;
                for (size_t k = 0; k < layers[i - 1].neurons.size(); k++)
                {
                    sum += layers[i - 1].neurons[k].output * layers[i].neurons[j].weights[k]; // sum of the weights multiplied by the output of the previous layer
                }
                layers[i].neurons[j].output = sigmoid(sum); // sigmoid activation function
            }
        }
    }

    // Backpropagation
    void backpropagation(const vector<double> &target)
    {
        // Calculate output layer deltas
        for (size_t i = 0; i < layers.back().neurons.size(); i++)
        {
            double output = layers.back().neurons[i].output;                                    // output of the neuron
            layers.back().neurons[i].delta = (target[i] - output) * sigmoid_derivative(output); // delta = (target - output) * derivative of the sigmoid function
        }

        // Calculate hidden layer deltas
        for (int i = layers.size() - 2; i > 0; i--)
        {
            for (size_t j = 0; j < layers[i].neurons.size(); j++)
            {
                double output = layers[i].neurons[j].output; // output of the neuron
                double sum = 0.0;
                for (size_t k = 0; k < layers[i + 1].neurons.size(); k++)
                {
                    sum += layers[i + 1].neurons[k].weights[j] * layers[i + 1].neurons[k].delta; // sum of the weights multiplied by the delta of the next layer
                }
                layers[i].neurons[j].delta = sum * sigmoid_derivative(output); // delta = sum * derivative of the sigmoid function
            }
        }

        // Update weights
        for (size_t i = 1; i < layers.size(); i++)
        {
            for (size_t j = 0; j < layers[i].neurons.size(); j++)
            {
                for (size_t k = 0; k < layers[i - 1].neurons.size(); k++)
                {
                    layers[i].neurons[j].weights[k] += eta * layers[i].neurons[j].delta * layers[i - 1].neurons[k].output; // weight = weight + learning rate * delta * output
                }
            }
        }
    }

    // Predict
    vector<double> predict(const vector<double> &input)
    {
        feed_forward(input);
        vector<double> result_of_prediction;
        for (const auto &neuron : layers.back().neurons)
        {
            result_of_prediction.push_back(neuron.output);
        }
        return result_of_prediction;
    }
};

// Load MNIST dataset for training
vector<pair<vector<double>, vector<double>>> load_mnist_train()
{
    vector<pair<vector<double>, vector<double>>> training_data;

    ifstream file("mnist_train.csv");
    string line;
    // Skip the first line since it contains column headers
    getline(file, line);
    while (getline(file, line))
    {
        istringstream iss(line);
        vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};

        vector<double> input;
        // Skip the first token which is the label
        for (size_t i = 1; i < tokens.size(); i++)
        {
            input.push_back(stod(tokens[i]) / 255.0); // Normalize the input to be between 0 and 1
        }

        vector<double> label(10, 0.0); // 10 classes
        label[stoi(tokens[0])] = 1.0;  // One-hot encoding

        training_data.push_back({input, label});
    }

    return training_data;
}

// Load MNIST dataset for testing
vector<pair<vector<double>, int>> load_mnist_test()
{
    vector<pair<vector<double>, int>> test_data;

    ifstream file("mnist_test.csv");
    string line;
    // Skip the first line since it contains column headers
    getline(file, line);
    while (getline(file, line))
    {
        istringstream iss(line);
        vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};

        vector<double> input;
        // Skip the first token which is the label
        for (size_t i = 1; i < tokens.size(); i++)
        {
            input.push_back(stod(tokens[i]) / 255.0); // Normalize the input to be between 0 and 1
        }

        int label = stoi(tokens[0]); // Label is in the first column
        test_data.push_back({input, label});
    }

    return test_data;
}

int main()
{
    // Neural network parameters
    vector<int> sizes = {784, 30, 10}; // 784 input neurons, 16 hidden neurons, 16 hidden neurons, 10 output neurons
    double learning_rate = 0.1;
    int num_epochs = 30;
    int seed = 42;

    // Create the neural network
    Network net(sizes, learning_rate);

    // Load the MNIST dataset
    vector<pair<vector<double>, vector<double>>> training_data = load_mnist_train();
    vector<pair<vector<double>, int>> test_data = load_mnist_test();

    // Train the neural network
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        // Shuffle the training data
        shuffle(training_data.begin(), training_data.end(), default_random_engine(seed));

        // Train the network
        for (const auto &data : training_data)
        {
            net.feed_forward(data.first);
            net.backpropagation(data.second);
        }

        // Evaluate the performance of the network
        int num_correct = 0;
        for (const auto &data : test_data)
        {
            vector<double> prediction = net.predict(data.first);
            int predicted_label = distance(prediction.begin(), max_element(prediction.begin(), prediction.end()));
            if (predicted_label == data.second)
            {
                num_correct++;
            }
        }

        cout << "Epoch " << epoch << ": " << num_correct << " / " << test_data.size() << " correct" << endl;
    }

    return 0;
}