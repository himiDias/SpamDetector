import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox




pd.options.display.float_format = "{:.1f}".format

print("=======================================START===================================================")

print('Packages imported')
 
# creates dataframe from csv file of sample spam dataset
dataframe = pd.read_csv(r'C:\Users\Himanshu-Desk\Documents\A-Level Project\firsttestdata.csv')


# defines the structure of the neural network
listneurons =  [12,14,14,12,1]

#function for calculating the sigmoid of activation
def sigmoid(z):
    return 1/(1+num.exp(-z))

#function for calcuating derivatice of Loss with respect to the output
def derivativeloss_output(output,label):
    return -((label*(1/output)) + (1-label)*(1/(1-output)))

#function for calculating the derivative of sigmoig with respect to the activation
def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

#function for calculating the binary cross entropy loss
def logloss(output,label):
    loss_ = -(num.dot(label,num.log(output))+ num.dot((1-label),num.log(1-output)))

    return loss_


#Defines the dataframe class
class Dataframe:

    #Intitialises an object of the class 
    def __init__(self,dataframe):
        self.df = dataframe.iloc[:,:]
        print("Dataframe object created")
    
    #Method for splitting the dataframe into 4 dataframes 
    def split(self,splitvalue):

        #shuffles the data into random order
        shuffleddata = self.df.sample(frac = 1.0 ,random_state = 1)

        #splits dataframe into train data and test data
        traindata = shuffleddata.sample(frac = splitvalue)
        testdata = shuffleddata.drop(traindata.index)

        #splits train and test data into 2 dataframes containing the sample features
        train_features = traindata.iloc[:,3:]
        test_features = testdata.iloc[:,3:]

        #splits train and test data into 2 dataframes containing the sample labels
        train_labels = traindata.iloc[:,2:3]
        test_labels = testdata.iloc[:,2:3]

        #converts the 2 labels dataframes into 2 arrays 
        train_labels = train_labels['labelbin'].to_numpy()
        test_labels = test_labels['labelbin'].to_numpy()
        feature_matrix = []

        #converts the 2 features dataframes into two 2-Dimensional arrays
        for i in train_features.columns:
            arr1 = num.array(train_features.loc[:,i])
            feature_matrix.append(arr1)
        train_features = num.transpose(feature_matrix)
        feature_matrix = []
        for j in test_features.columns:
            arr2 = num.array(test_features.loc[:,j])
            feature_matrix.append(arr2)
        test_features = num.transpose(feature_matrix)

        #returns the 4 arrays
        return train_features,test_features,train_labels,test_labels



#Defines the neural network class
class NeuralNetwork:

    #Intitialises and object of the class 
    def __init__(self,list_neurons):
        self.num_layers = len(list_neurons)
        self.neurons = list_neurons

        #Creates random values for intial weights and biases
        self.biases = [num.random.randn(i,1) for i in list_neurons[1:]]
        self.weights = [num.random.randn(i,j) for i,j in zip(list_neurons[1:],list_neurons[:-1])]
        print("Neural Network object created")

    #Method for running a sample through the neural network
    def feedforward(self,list_neurons,input,label):
        self.z_values = [num.zeros(b.shape) for b in self.biases]
        self.activations = [num.zeros(b.shape) for b in self.biases]
        self.activations.insert(0,input)

        #feedforward through each layer in network
        for x in range (len(list_neurons)-1):
            output = []
            for w,b in zip(self.weights[x],self.biases[x]):
                z = num.dot(w,input) + b
                self.z_values[x][num.where(self.biases[x] == b)] = z
                self.activations[x+1][num.where(self.biases[x] == b)] = sigmoid(z)
                output.append(sigmoid(z))
            input = output
        #returns the loss and the output
        return logloss(input[0],label),input[0]
    
    #Method for backpropogating through the neural network
    def backpropagate(self,list_neurons,label):
        div_biases = [num.zeros(b.shape) for b in self.biases]
        div_weights = [num.zeros(w.shape) for w in self.weights]
        #calculates delta of final layer
        delta = derivativeloss_output(self.activations[-1],label) * sigmoid_deriv(self.z_values[-1])

        #calculates bias and weight derivatives of final layer
        div_biases[-1] = delta
        div_weights[-1] = num.transpose(num.dot(self.activations[-2],delta))

        #performs backpropogation through each layer
        for l in range (2,len(list_neurons)):
            div_s = []
            z = self.z_values[-l]
            div_s = sigmoid_deriv(z)
            delta = num.dot(num.transpose(self.weights[-l+1]),delta) * div_s
            div_biases[-l] = delta
            div_weights[-l] = num.dot(delta,num.transpose(self.activations[-l-1]))

        #return array of bias and weight derivatives
        return div_biases,div_weights
    
    #Method for carrying out gradient descent
    def gradient_descent(self,learning_rate,listNeurons,label):

        #obtains derivatives by calling backpropagate
        bias_divs,weight_divs = self.backpropagate(listNeurons,label)

        #adjusts weights and biases of each neuron for each layer
        for l in range (1,len(listNeurons)):
            adjusted_dw = weight_divs[-l] * learning_rate
            adjusted_db = bias_divs[-l] * learning_rate
            self.weights[-l] = num.dot(label,self.weights[-l] - adjusted_dw) + num.dot(label - 1,self.weights[-l] + adjusted_dw)
            self.biases[-l] = num.dot(label,self.biases[-l] - adjusted_db) + num.dot(label - 1,self.biases[-l] + adjusted_db)
    
    #Method for training the neural network
    def train(self,netstructure,traininputdf,testinputdf,labeldf,testlabeldf,lossb,learningrate):
        average_train_loss = 10
        average_test_loss = 10
        epoch = 0
        y = []
        y1 = []
        x = []

        #performs gradient descent until a satisfactory training loss is met
        while average_train_loss > lossb:
            
            train_losses = []
            #manipulates inputs into 2D array before passing into neural network
            for i in range(0,len(traininputdf)):
                inputfix = num.arange(len(traininputdf[i])).reshape(len(traininputdf[i]),1)
                for j in range (0,len(traininputdf[i])):
                    inputfix[j] = traininputdf[i][j]

                #passes the fixed inputs into the neural network
                train_loss,train_output = nn.feedforward(netstructure,inputfix,labeldf[i])

                #stores the loss 
                train_losses.append(train_loss)

                #performs gradient descent
                nn.gradient_descent(learningrate,netstructure,labeldf[i])

            #calculates the average training loss
            average_train_loss = num.average(train_losses)

            test_losses = []

            #manipulates inputs like before
            for k in range(0,len(testinputdf)):
                testinputfix = num.arange(len(testinputdf[k])).reshape(len(testinputdf[k]),1)
                for l in range(0,len(testinputdf[k])):
                    testinputfix[l] = testinputdf[k][l]

                #passes inputs into the neural network
                testloss,testoutput = nn.feedforward(netstructure,testinputfix,testlabeldf[k])

                #stores the loss
                test_losses.append(testloss)

            #calculates average test loss
            average_test_loss = num.average(test_losses)

            #stores average losses and epoch number in 3 different arrays
            y.append(average_train_loss)
            y1.append(average_test_loss)
            x.append(epoch)

            #outputs epoch and losses
            print("EPOCH: ", epoch + 1," ¦¦ AVG TRAIN LOSS: ",average_train_loss, "¦¦ AVG TEST LOSS: ",average_test_loss)
            epoch += 1
        
        #returns the 3 arrays and the trained weights and biases
        return y,x,y1,self.weights,self.biases
    
    #Method for predicting 
    def predict(self,input,netstruc):
         
         #perfoms feedforward
         for x in range (len(netstruc)-1):
            output = []
            for w,b in zip(self.weights[x],self.biases[x]):
                z = num.dot(w,input) + b
                output.append(sigmoid(z))
            input = output

        #returns prediction depending on output
         if input[0] > 0.5:
            return input,"Prediction: SPAM"
         else:
            return input,"Prediction: NOT SPAM"



#creating neural network object
nn = NeuralNetwork(listneurons)

#creating dataframe object
data = Dataframe(dataframe)

#splitting dataframe
tr_f,te_f,tr_l,te_l=data.split(0.75)


#training neural network 
y_val,x_val,y1_val,final_weights,final_biases=nn.train(listneurons,tr_f,te_f,tr_l,te_l,0.2,0.05)
 
#plot a graph of training and test loss against epoch
plt.plot(x_val,y_val)
plt.plot(x_val,y1_val)

#labels axis
plt.xlabel("Epoch")
plt.ylabel("Loss")

#shows graph
plt.show()

#Defines EmailGUI class
class EmailGUI:

    #Intialises the GUI 
    def __init__(self, Root):
        self.Root = Root
        Root.title("Email GUI")

        #Label and Text Box
        self.email_label = tk.Label(Root, text="Email:")
        self.email_label.pack()
        self.email_textbox = tk.Text(Root, height=24, font=("Arial", 12))
        self.email_textbox.pack(pady=10 , padx = 10)

        #Submit button
        self.submit_button = tk.Button(Root, text="Submit", command=self.save_email)
        self.submit_button.place(x=450, y = 550)

        #Clear button
        self.clear_button = tk.Button(Root, text="Clear", command=self.clear_text)
        self.clear_button.place(x=320, y = 550)

    #Method for saving contents of textbox
    def save_email(self):
        email = self.email_textbox.get("1.0", tk.END).strip()
        listofexpressions = ["£","won",".com","click","hot","xxx","$","!","gamble","%","security","alert"]
        inputs =[]

        #Counts occurences of trigger words in the email
        for i in listofexpressions:
            x = email.lower().count(i)

            #stores occurences in array
            inputs.append(x)
        print(inputs)

        #passes array as inputs to trained neural network
        #outputs an info window showing result and prediction
        messagebox.showinfo(title="Prediction", message=("Probabilty :",nn.predict(inputs,listneurons)),)
    
    #Method for clearing contents of textbox
    def clear_text(self):
        self.email_textbox.delete("1.0",tk.END)


#Creates main widget/window
root = tk.Tk()

#resizes
root.geometry("800x600")

#creates EmailGUI object
email_gui = EmailGUI(root)

#runs until user quits
root.mainloop()