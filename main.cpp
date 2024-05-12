#include <iostream> // Standard input/output
#include <vector>   // Vectors
#include <cmath>    // Math functions
#include <cstdlib>  // Random number generation
#include <ctime>    // Time functions

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
