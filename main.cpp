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
};