# TO DO Install numpy eventually
#import numpy as np


# nueron variables
inputs = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1.0],
		   [0.5,-0.91,0.26,-0.5],
		   [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]




layer_outputs = [] # output of current layer

# equation for guessing := inputs[n] * weights[n] + inputs[n+1] * weights[n+1] ..... + bias
# loop computes dot product for neural network
# output_layer = np.dot(weights,inputs) + biases
# zip combines lists together takes inputs and weights and makes [[1,[0.2,0.8,-0.5,1.0]],....,[2.5[-0.26,-0.27,0.17,0.87]]]
for neuron_weight, neuron_bias in zip(weights,biases):
	neuron_output = 0 
	for n_input, weight in zip(inputs, neuron_weight):
		neuron_output += n_input*weight
	neuron_output += neuron_bias
	layer_outputs.append(neuron_output)

print(layer_outputs)
