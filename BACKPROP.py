from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
sentence = input("Please enter the text: ")
print("The input string: ", sentence)
corpora=sentence
wordtoken_test=word_tokenize(sentence)
print("Tokenization of testing data: ",wordtoken_test)


from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
post_test=nltk.pos_tag(wordtoken_test)
print("Parts of speech tagging of testing data:",post_test)

feature_vector=[]
connectives=[]
who=[]
which=[]
when=[]
what=[]
where=[]
how=[]
pos_tagging2=[]

#CATEGORIZE THE TESTING DATA
for i in range(0,len(wordtoken_test)):
    if 'ADJ' in post_test[i][1]:
        feature_vector.append(1)
    elif 'ADV' in post_test[i][1]:
        feature_vector.append(1)
    elif 'CNG' in post_test[i][1]:
        feature_vector.append(1)
    elif 'DET' in post_test[i][1]:
        feature_vector.append(1)
    elif 'EX' in post_test[i][1]:
        feature_vector.append(1)
    elif 'MD' in post_test[i][1]:
        feature_vector.append(1)
    elif 'N' in post_test[i][1]:
        feature_vector.append(0)
    elif 'NP' in post_test[i][1]:
        feature_vector.append(0)
    elif 'NN' in post_test[i][1]:
        feature_vector.append(0)
    elif 'NUM' in post_test[i][1]:
        feature_vector.append(0)
    elif 'PRP' in post_test[i][1]:
        feature_vector.append(0)
    elif 'P' in post_test[i][1]:
        feature_vector.append(1)
    elif 'TO' in post_test[i][1]:
        feature_vector.append(1)
    elif 'UH' in post_test[i][1]:
        feature_vector.append(1)
    elif 'V' in post_test[i][1]:
        continue
    elif 'VD' in post_test[i][1]:
        continue
    elif 'VG' in post_test[i][1]:
        continue
    elif 'VN' in post_test[i][1]:
        continue
    elif 'WH' in post_test[i][1]:
        feature_vector.append(1)
    else:
        continue

print(feature_vector)
import numpy as np


#TRANSFER FUNCTIONS
def sgm(x,Derivative=False):
    if not Derivative:
        return 1/(1+np.exp(-x))
    else:
        out = sgm(x)
        return out*(1.0 - out)

def linear(x,Derivative=False):
    if not Derivative:
        return x
    else:
        return 1.0

def gaussian(x,Derivative=False):
    if not Derivative:
        return np.exp(-x**2)
    else:
        return -2*x*np.exp(-x**2)

def sgm(x,Derivative=False):
    if not Derivative:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2

#CLASS DECLARATION
class BackPropagationNetwork:
    """A Back Propagation Network"""

    #CLASS MEMBERS
    layerCount = 0
    shape = None
    weights=[]
    tFuncs = []


    #CLASS METHODS
    def __init__(self, layerSize, layerFunctions= None):
        """Initialize the Network"""

        #LAYER INFO
        self.layerCount = len(layerSize)-1
        self.shape = layerSize

        if layerFunctions is None:
            lFuncs = []
            for i in range(self.layerCount):
                if i == self.layerCount - 1:
                    lFuncs.append(linear)
                else:
                    lFuncs.append(sgm)
        else:
            if len(layerSize) != len(layerFunctions):
                raise ValueError("Incompitable Transfer Functions.")
            elif layerFunctions[0] is not None:
                raise ValueError("Input layer cannot have transfer function.")
            else:
                lFuncs = layerFunctions[1:]
                
        self.tFuncs = lFuncs
        #DATA FROM LAST RUN
        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []
        
        
        #CREATE THE WEIGHT ARRAYS
        for (l1,l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.1, size= (l2,l1+1)))
            self._previousWeightDelta.append(np.zeros((l2,l1+1)))
    #RUN METHOD
    def Run(self, inputs):
        "Run the Network based on the Input Data"""
        lnCases = inputs.shape[0]
        #Clear out the previous intermediate value lists
        self._layerInput = []
        self._layerOutput = []
        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([inputs.T, np.ones([1, lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.tFuncs[index](layerInput))
            
        return self._layerOutput[-1].T


    def TrainEpoch(self, inputs, target, trainingRate= 0.2, momentum=0.5):
        """This method trains the network for one epoch"""

        delta = []
        lnCases = inputs.shape[0]
        self.Run(inputs)

        #CALCULATE DELTAS
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                #COMPARE TO THE TARGET VALUES
                output_delta = self._layerOutput[index]-target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.tFuncs[index](self._layerInput[index], True))
            
        
            else:
                #COMPARE TO THE FOLLOWING LAYER DELTA
                delta_pullback = self.weights[index+1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :]*self.tFuncs[index](self._layerInput[index],True))
                
                
        #COMPUTE WEIGHT DELTAS
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1- index

            if index == 0:
                layerOutput = np.vstack([inputs.T, np.ones([1, lnCases])])
            else:
                layerOutput = np.vstack([self._layerOutput[index-1], np.ones([1, self._layerOutput[index-1].shape[1]])])


            curWeightDelta=np.sum(layerOutput[None,:,:].transpose(2,0,1)*delta[delta_index][None,:,:].transpose(2,1,0), axis=0)                                      
            weightDelta = trainingRate*curWeightDelta+momentum*self._previousWeightDelta[index]
            self.weights[index] -= weightDelta
            self._previousWeightDelta[index] = weightDelta

        return error                      
                
    
        
    
if __name__== "__main__":
    
    #print(bpn.shape)
    #print(bpn.weights)
    lFuncs= [None, sgm, sgm]
    bpn=BackPropagationNetwork((len(feature_vector), 3, 2), lFuncs)
    target=[0,1]
    lvInput = np.array([feature_vector])
    lvTarget= np.array([[0,1]])
    lnMax= 1000000

    lnErr=1e-5
    for i in range(lnMax+1):
        err=bpn.TrainEpoch(lvInput, lvTarget, momentum = 0.7)
        if i%100 == 0:
            print("Iteration: ",i,"Error: ",err)
        if err <= lnErr:
            print("Minimum error reached at iteration: ",i)
            break

    #DISPALY OUTPUT
    lvOutput= bpn.Run(lvInput)
    for i in range(lvInput.shape[0]):
        print("input: ",lvInput[i],"Output: ",lvOutput[i])
            
    
        

