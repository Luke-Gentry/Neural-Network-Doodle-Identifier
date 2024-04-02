import time
import numpy as np
import math
import matplotlib as plt

############################### ACTIVATION FUNCTIONS ###############################

def sigmoid(x):
    sigmoidal_array = np.zeros(len(x))
    for i in range(len(x)):
        sigmoidal_array[i] = 1 / (1 + np.exp(-(x[i])))
    return sigmoidal_array

def sigmoid_one(x):
    return 1 / (1 + np.exp(-(x)))

def Re_LU(x):
    return np.maximum(0, x)

def SoftMax(outputs):
    e = math.e
    exp_values = (e ** outputs)
    norm_base = sum(exp_values)
    norm_values = np.around(( exp_values / norm_base), rounding_const)
    return norm_values


############################### LAYER CLASS ###############################


class Layer:
    def __init__(self, curr_neurons, output_neurons, name): # Layer Initialization fills weights and biases with random starting value to later be optimized
        self.weights = .1 * np.random.randn(curr_neurons, output_neurons)
        self.gradients = np.zeros((curr_neurons, output_neurons))
        self.cost_due_to_next = 0
        self.biases = .1 * np.random.randn(1, output_neurons)
        self.bias_gradients = np.zeros((1, output_neurons))
        self.name = name
        self.curr_neurons = curr_neurons
        self.output_neurons = output_neurons

    def forward(self, inputs):
        self.values = inputs
        self.preactivation_outputs = ((np.dot(inputs, self.weights) + self.biases))
        self.preactivation_outputs = self.preactivation_outputs[0]

    def activation(self):
        self.outputs = np.around(Re_LU(self.preactivation_outputs), rounding_const)

    def soft_max(self):
        self.outputs = np.around(SoftMax(self.preactivation_outputs), rounding_const)

    def reset_gradients(self):
        self.gradients = np.zeros((self.curr_neurons, self.output_neurons))
        self.bias_gradients = np.zeros((1, self.output_neurons))
        
    def update_weights(self, learning_rate, batch_size):
        self.weights += -self.gradients * learning_rate
        self.biases += -self.bias_gradients * learning_rate

        #if (self.name == "Hidden Layer 2"):
        #    print(self.gradients)

        self.bias_gradients = np.zeros((1, self.output_neurons))
        self.gradients = np.zeros((self.curr_neurons, self.output_neurons))


    def print(self):
        print(self.name)
        print("Values:")
        print(self.values)
        if (self.name != "Output Layer"):
            print("Weights:")
            print(self.weights)
            print("Biases:")
            print(self.biases)
            print("Preactivation Outputs:")
            print(self.preactivation_outputs)
            print("Outputs:")
            print(self.outputs)
        

layers=[]
rounding_const = 8

def InitializeNeuralNetworkLayers(setup_list):
    # [3, 4, 3, 0]
    setup_list.append(0)
    for i in range(len(setup_list)-1):
        name = ""
        if (i == 0):
            name = "Input Layer"
        elif (i == len(setup_list) - 2):
            name = "Output Layer"
        else:
            name = "Hidden Layer " + (str(i))
        layers.append(Layer(setup_list[i], setup_list[i+1], name))


def ForwardPropogation(inputs): # inputs is a 2d array of inputs, each row is an input
    for input in inputs:
        for i in range(len(layers)): # for first layer calculate new values based on layer's weights and biases from input
            if (i == 0):
                layers[i].forward(input)
                if (len(layers) != 2):
                    layers[i].activation() # send values through activation function to add non-linearity
            else:
                if (i != len(layers) - 1):
                    layers[i].forward(layers[i-1].outputs) # for all other layers calculate new values based on layer's weights and biases from last layer outputs
                    if (i != len(layers) - 2):
                        layers[i].activation()
                    else:
                        layers[i].soft_max()
                else:
                   layers[i].values = layers[i-1].outputs
        return layers[len(layers) - 1].values
    



def CalculateLoss (prediction_values, label): # calculate cost (how wrong the neural network was based on actual vs expected output
    target_values = np.zeros(10)
    target_values[label] = 1
    cost = 0

    for i in range(len(prediction_values)):
        cost += (np.square(prediction_values[i] - target_values[i]))
    cost = np.around(cost, rounding_const)
    return cost



def CalculateCostWRTNeuron (layer_n, target_values, output_layer_index): # calculate how much of the total cost was due to a specific nueron
    if layer_n == 0:
        cost = 2 * (layers[len(layers) - 1].values[output_layer_index] - target_values[output_layer_index])
        return cost
    else: # uses chain rule to determine how much of an effect the nueron has on the output
        recursive_cost = 0
        next_layer = layers[len(layers) - 1 - layer_n]
        for next_next_index in range(len(next_layer.outputs)):
            next_neuron_weight = next_layer.weights[output_layer_index][next_next_index]
            derivative = 0
            if (layer_n + 2 >= len(layers)):
                derivative = next_layer.outputs[next_next_index] * (1 - next_layer.outputs[next_next_index])
            else:
                derivative = next_layer.outputs[next_next_index] * (1 - next_layer.outputs[next_next_index])
            cost_due_to_next_nueron = CalculateCostWRTNeuron(layer_n - 1, target_values, next_next_index)
            recursive_cost += np.around(next_neuron_weight * derivative * cost_due_to_next_nueron, rounding_const)
        return recursive_cost



def CalculateGradient (curr_layer_index, output_layer_index, layer_index, target_values): # calculates "direction" of the cost function based on the change in weights

    curr_layer = layers[len(layers) - 2 - layer_index]
    source_neuron = curr_layer.values[curr_layer_index]
    derivative = curr_layer.outputs[output_layer_index] * (1 - curr_layer.outputs[output_layer_index])
    cost_due_to_next_nueron = CalculateCostWRTNeuron(layer_index, target_values, output_layer_index)
    gradient = np.around(source_neuron * derivative * cost_due_to_next_nueron, rounding_const)

    return gradient


def CalculateGradientBias (curr_layer_index, output_layer_index, layer_index, target_values): # calculates "direction" of the cost function based on the change in biases

    curr_layer = layers[len(layers) - 2 - layer_index]

    source_neuron = 1
    derivative = curr_layer.outputs[output_layer_index] * (1 - curr_layer.outputs[output_layer_index])
    cost_due_to_next_nueron = CalculateCostWRTNeuron(layer_index, target_values, output_layer_index)

    gradient = np.around(source_neuron * derivative * cost_due_to_next_nueron, rounding_const)

    return gradient



def BackPropogationMatrixReLU (label, batch_size): # stores in gradients variable the amount and direction in which wiehgt and bias should change to reduce the cost function
    target_values = np.zeros(len(layers[len(layers) - 1].values))
    target_values[label] = 1
    for k in range(len(layers) - 1):
        if k == 0: 
            curr_layer = layers[len(layers) - 2]
            curr_layer_values = curr_layer.values
            next_layer_values = layers[len(layers) - 1].values
            curr_layer.cost_due_to_next = (2 * ((next_layer_values) - target_values))
            deriv = (np.multiply(next_layer_values, (1 - next_layer_values)))
            temp_array = np.multiply(deriv, curr_layer.cost_due_to_next)
            curr_layer.gradients += (1/batch_size) * np.dot(np.atleast_2d(curr_layer_values).T, np.atleast_2d(temp_array))
            curr_layer.bias_gradients += (1/batch_size) * np.atleast_2d(temp_array)


        else:
            curr_layer = layers[len(layers) - 2 - k]
            next_layer = layers[len(layers) - 1 - k]
            next_next_layer = layers[len(layers) - k]
            curr_layer_values = curr_layer.values
            next_layer_values = next_layer.values
            next_next_layer_values = next_next_layer.values

            curr_deriv = (next_layer_values > 0).astype(int)
            
            if next_next_layer.name == "Output Layer":
                next_deriv = (np.multiply(next_next_layer_values, (1 - next_next_layer_values)))
            else:
                next_deriv = (next_next_layer_values > 0).astype(int)
            next_cost = next_layer.cost_due_to_next
            sum_cost = np.dot(next_layer.weights, np.multiply(next_deriv, next_cost))
            temp_array = np.multiply(curr_deriv, sum_cost)

            curr_layer.cost_due_to_next = sum_cost
            curr_layer.gradients += (1/batch_size) * np.multiply(np.atleast_2d(curr_layer_values).T, np.atleast_2d(temp_array))
            curr_layer.bias_gradients += (1/batch_size) * np.atleast_2d(temp_array)



def UpdateWeights(batch_size): # updates weights based on output of BackPropogationMatrix function
    for i in range(len(layers) - 1):
        curr_layer = layers[i]
        curr_layer.weights += -curr_layer.gradients
        curr_layer.biases += -curr_layer.bias_gradients
            
    
    
def RunNeuralNetwork(inputs, labels, total_runs, batch_size, learning_rate):
    print(total_runs // batch_size)
    print(batch_size)
    b = False
    batch_cost = 0
    for j in range(total_runs // batch_size):  
        for i in range(batch_size):
            r = np.random.randint(0, len(inputs))

            input = inputs[r].flatten() / 255
            layer_output = ForwardPropogation([input])
            if (math.isnan(layer_output[0])):
                b = True
                print("BROKEN")
                break
            batch_cost += CalculateLoss(layer_output, int(labels[r]))


            BackPropogationMatrixReLU(int(labels[r]), batch_size)
            if (math.isnan(CalculateLoss(layer_output, int(labels[r])))):
                b = True
                print("BROKEN")
                break
                
        if b == True:
            break

        for k in range(len(layers) - 1):
            layers[k].update_weights(learning_rate, batch_size)
        print(str(batch_cost) + " -- " + str(j))
        batch_cost = 0


def TestNeuralNetwork(inputs, labels):

    total_correct = 0
    total_wrong = 0

    doodle_names = ["Angel", "Bus", "Traffic Light", "Eiffel Tower", "Skull", "Spider"]
    track = np.zeros(len(doodle_names))
    total = np.zeros(len(doodle_names))
    
    for i in range(len(inputs)):
        input = inputs[i].flatten() / 255
        layer_output = ForwardPropogation([input])

        CalculateLoss(layer_output, int(labels[i]))
        t = np.argmax(layer_output)
        if ((t == int(labels[i]))):
            track[t] += 1
        total[int(labels[i])] += 1



        if layer_output[labels[i]] == np.max(layer_output):
            total_correct += 1
        else:
            total_wrong += 1
        
    print(str(total_correct + total_wrong) + " --> TOTAL")
    print(str(total_correct) + " --> TOTAL CORRECT")
    print(str(total_wrong) + " --> TOTAL WRONG")
    print(str(total_correct / (total_correct + total_wrong)) + " --> ACCURACY")
    print(track)
    print(total)
    for i in range(len(doodle_names)):
        print(doodle_names[i] + " --> " + str(track[i]/total[i]))
        



def RunNeuralNetworkTest(input):
    layer_output = ForwardPropogation(input)
    return layer_output

