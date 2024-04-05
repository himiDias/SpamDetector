import pandas as pd
import numpy as num
import matplotlib as plt

pd.options.display.float_format = "{:.1f}".format

print('Packages imported')


# creates dataframe from csv file of sample spam dataset
dataframe = pd.read_csv(r'C:\Users\Himanshu-Desk\Documents\A-Level Project\firsttestdata.csv')
#print(dataframe)

feature_matrix = []
#for i in dataframe.columns:
    #array = num.array(dataframe.loc[:,i])
    #print(array)
    #feature_matrix.append(array)


#structure of network, this shows there are 3 layers (input,hidden,output) with the input layer having 1 neuron (1 feature used), 4 neurons in hidden layer and 1 output
list_neurons = [1,4,1]
n_biases = []
n_weights = []
#as this particular input only uses one feature of which one sample at a time, the input in just one element
inputs = [4]


def sigmoid(z):
    return 1/(1 + num.exp(-z))

    
def initial_biases(layers):
    n_biases = [num.random.randn(i,1) for i in layers[1:]]
    return n_biases

def initial_weights(layers):
    n_weights = [num.random.rand(i,j) for i,j in zip(layers[1:],layers[:-1])]
    return n_weights

def feedforward(weights,biases,input,layer):
    print("---FEEDFORWARD---")
    output = []
    zvalues = []
    for w,b in zip(weights[layer],biases[layer]):
        z = num.dot(w,input) + b
        print("z values: ",z)
        zvalues.append(z)
        output.append(sigmoid(z))
        print("activations: ",output)
    print("--END--")
    return zvalues,output

# Binary Cross entropy loss   
def loss_function(output,label):
    loss = -(num.dot(label,num.log(output)) + num.dot((1-label),num.log(1-output)))
    return loss

#Backpropagation functions
def backpropagate(weights,biases,output,label,layers):
    div_biases = [num.zeros(b.shape) for b in biases]
    div_weights = [num.zeros(w.shape) for w in weights]
     
    
    errorL = derivativeloss_output(activations[-1][0],label) * sigmoid_deriv(z_values[-1][0])
    print("ERROR : ",errorL)

    div_biases[-1] = errorL
            

    div_weights[-1] = num.dot(activations[-2],errorL)
    print("dC/db : ",div_biases)
    print("dC/dw : ", div_weights)

    for l in range (2,layers):
        a_s = []
        z = z_values[-l]
        print(z)
        for x in z:
            a_s.append(sigmoid_deriv(x))
        print(a_s)
        errorL = num.dot(num.transpose(weights[-l+1]),errorL) 
        print("ERROR: ",errorL)
        print(num.transpose(errorL))
        div_biases[-l] = num.transpose(errorL)
        div_weights[-l] = num.dot(num.transpose(activations[-l-1]),errorL)
    print(div_weights[0])
    print("dC/db : ",div_biases)
    print("dC/dw : ",div_weights)



def derivativeloss_output(output,label):
    return -((label*(1/output)) + (1-label)*(1/(1-output)))

def derivativeloss_bias(error):
    return error

def derivativeloss_weights(output,error):

    return 

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

def errorL (dCdA,sigdiv):
    return dCdA * sigdiv



weights = initial_weights(list_neurons)
biases = initial_biases(list_neurons)
z_values = []
activations = [inputs]

print("INITIAL WEIGHTS: ",weights)
print("INITIAL BIASES: ",biases)

for x in range (len(list_neurons) - 1):
    zs,new_input = feedforward(weights,biases,inputs,x)
    z_values.append(zs)
    activations.append(new_input)
    print ("OUTPUT", x,": ",new_input)
    inputs = new_input

print("Z VALUES: ",z_values)
print("ACTIVATIONS: ",activations)

#print(z_values[0])
 
#print(loss_function(inputs,1))
#print(derivativeloss_output(inputs,1))

#print(errorL((derivativeloss_output(inputs,1)),sigmoid_deriv(z_values[-1])))

#backpropagate(weights,biases,inputs,1,len(list_neurons))