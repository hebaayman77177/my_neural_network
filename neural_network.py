import numpy as np
import scipy.special
import matplotlib.pyplot
class neural_network:


       # initialize the neural network
       def __init__(self,input_nodes,
       	hidden_nodes,output_nodes,learning_rate):
       	
       	#set number of nodes in each input ,hidden, output layer
       	self.inodes=input_nodes
       	self.hnodes=hidden_nodes
       	self.onodes=output_nodes

       	#set learning rate
       	self.lr=learning_rate
       	

       	#initiate the weights
       	self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
       	self.who=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))


       	#set the activation function to the sigmoid function
       	self.activation_function=lambda x:scipy.special.expit(x)

       	






       # train the neural network
       #take one input and one output ,then compute the error then update the weights
       def train(self, inputs_list, targets_list):
       
       	# convert inputs list to 2d array

           inputs = np.array(inputs_list, ndmin=2).T
           targets = np.array(targets_list, ndmin=2).T

           #calculate the signals into the hidden layer
           hidden_inputs=np.dot(self.wih,inputs)

           #calculate the signal out from hidden layer
           hidden_outputs=self.activation_function(hidden_inputs)

           #calculate the signal into the final output layer
           final_inputs=np.dot(self.who,hidden_outputs)

           #calculate the signal out from final layer
           final_outputs=self.activation_function(final_inputs)

           #calculate the error 
           output_errors = targets - final_outputs

           # hidden layer error is the output_errors, split by weights, recombined at hidden nodes 
           hidden_errors = np.dot(self.who.T,output_errors)

           # update the weights for the links between the hidden and output layers 
           self.who += self.lr *np.dot((output_errors * final_outputs *(1.0-final_outputs)) ,np.transpose(hidden_outputs))
       	

           # update the weights for the links between the input and hidden layers 
           self.wih +=self.lr *np.dot((hidden_errors * hidden_outputs *(1.0- hidden_outputs)) ,np.transpose(inputs))
           


       # query the neural network
       #give an input and produce the output from the input and weights
       def query(self,input_list):

       	#convert the input list to 2d array with 1 column and 3 raws
       	inputs=np.array(input_list, ndmin=2).T

       	#calculate the signals into the hidden layer
       	hidden_inputs=np.dot(self.wih,inputs)

       	#calculate the signal out from hidden layer
       	hidden_outputs=self.activation_function(hidden_inputs)

       	#calculate the signal into the final output layer
       	final_inputs=np.dot(self.who,hidden_outputs)

       	#calculate the signal out from final layer
       	final_outputs=self.activation_function(final_inputs)

       	return final_outputs


##########################################################################################
##########################################################################################
###################initiating the network



# number of input, hidden and output nodes 
input_nodes = 784 
hidden_nodes = 200 
output_nodes = 10 

# learning rate is 0.3 
learning_rate = 0.15




# create instance of neural network 
n = neural_network(input_nodes,hidden_nodes,output_nodes, learning_rate) 

##########################################################################################


###########################################################################################
###########################################################################333#############
###############training algorithm

# load the mnist training data CSV file into a list 
training_data_file = open("mnist_train.csv", 'r') 
training_data_list = training_data_file.readlines() 
training_data_file.close() 


# go through all records in the training data set
i=0 
epochs=1
for e in range(epochs):
      for record in training_data_list:
          print("ok",i,"in epoch",e)
          
          if (i in [43666]):
              continue

          i=i+1
          # split the record by the ',' commas 
          all_values = record.split(',') 
          # scale and shift the inputs 
          inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 
          
          # create the target output values (all 0.01, except the desired label which is 0.99) 
          targets = np.zeros(output_nodes) + 0.01 

          # all_values[0] is the target label for this record 
          targets[int(all_values[0])] = 0.99 
          n.train(inputs, targets) 
          

#######################################################################################
########################################################################################
##################testing




test_data_file = open("mnist_test.csv", 'r') 
test_data_list = test_data_file.readlines() 
test_data_file.close() 







c=0
for record in test_data_list:
      all_values=record.split(',')
      true_label=int(all_values[0])
      result=n.query((np.asfarray(all_values[1:]))/255*0.9+0.01)
      predected_label=np.argmax(result)
      print("#################################")
      print(predected_label)
      print(true_label)
      if(true_label == predected_label):
            c=c+1


print(c/len(test_data_list))











