#!/usr/bin/env python
# coding: utf-8

# In[1]:


# INSTALL LIBROSA
get_ipython().system('pip install librosa')


# In[2]:


# INSTALL TORCH
get_ipython().system('pip install torch')


# In[3]:


# IMPORT PACKAGES/MODULES
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import random

from IPython import display
import librosa
import librosa.display

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# SET SEED
# For experiment reproducibility
seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# In[5]:


# OPEN TRAINING FILE + TEST FILE
# Training file
infile = open("raw_data.pkl",'rb')
data, targets = pickle.load(infile)
print("Some info about the TRAINING targets:")
print("   Type of targets:", type(targets))
print("   Total number of targets:", len(targets))
print("   Target example:", targets[0])
print()
print("Some info about the TRAINING data:")
print("   Type of data:", type(data))
print("   Total number of data:", len(data))
print(".  Shape of the data:", data.shape)
print("   Data example:\n", data[0])
infile.close()

# Test file
testfile = open("test_data.pkl",'rb')
testdata = pickle.load(testfile)
print("\nSome info about the TEST data:")
print("   Type of data:", type(testdata))
print("   Length of the data:", len(testdata[0]))
print("   Length of one sequence:\n", len(testdata[0][1]))
testfile.close()

# Reference: https://docs.python.org/3/library/pickle.html


# In[6]:


# AUDIO SAMPLES
# Hearing several audio records
print('Audio playback')
display.display(display.Audio(data[21], rate = 11025))
display.display(display.Audio(data[22], rate = 11025))
display.display(display.Audio(data[23], rate = 11025))
display.display(display.Audio(data[24], rate = 11025))
display.display(display.Audio(data[25], rate = 11025))

# Reference: https://ipython.org/ipython-doc/3/api/generated/IPython.display.html


# ## START PRE-PROCESSING / FEATURE EXTRACTION

# In[7]:


# CREATE MFCC FEATURES OF THE TRAIN DATA 
features = []

for n in data:
    mfccs = librosa.feature.mfcc(n, n_mfcc = 22, sr = 11025)
    features.append(mfccs)

# Some info about the MFCC features
print("Total length of mfcc features:", len(features))
print("Shape of the features:", len(features))

# Reference: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
# Reference: https://mikesmales.medium.com/sound-classification-using-deep-learning-8bc2aa1990b7


# In[8]:


# SPLIT THE DATA INTO TRAIN + VAL 
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size = 0.1, random_state = 999, shuffle = False)

# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


# In[9]:


# FUNCTION TO MAKE PAIRS
def make_pairs(sequences, labels):
    # Define empty lists
    Seq0 = [] 
    Seq1 = []
    pairs = []

    lookup = max(np.unique(labels)) + 1
    index = [np.where(labels == i)[0] for i in range(0, lookup)]

    # Loop over all sequences
    for index_a in range(len(sequences)):
        current_seq = sequences[index_a]
        label = labels[index_a]
        
        for i in range(2):
            # Positive pair
            index_b = np.random.choice(index[label])
            pos_seq = sequences[index_b]
            Seq0.append([current_seq])
            Seq1.append([pos_seq])
            pairs.append([0])

            # Negative pair
            neg_index = np.where(labels != label)[0]
            neg_seq = sequences[np.random.choice(neg_index)]
            Seq0.append([current_seq])
            Seq1.append([neg_seq])
            pairs.append([1])

    return (np.array(Seq0), np.array(Seq1), np.array(pairs))

# Reference: https://www.pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/


# In[10]:


# CREATE POSITIVE / NEGATIVE PAIRS FOR TRAIN DATA
(Seq0, Seq1, labels) = make_pairs(X_train, y_train)


# In[11]:


# CONVERT THE TRAINING DATA TO A TRAINLOADER
Seq0 = torch.Tensor(Seq0)
Seq1 = torch.Tensor(Seq1)
labels = torch.Tensor(labels)
traindataset = TensorDataset(Seq0, Seq1, labels)
train_loader = DataLoader(traindataset, batch_size = 64, shuffle = True, num_workers = 0)


# # Define neural network 

# In[12]:


# CONTRASTIVE LOSS
class ContrastiveLoss(torch.nn.Module):
    # Contrastive loss function
    # Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
   
    def __init__(self, margin = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min = 0.0), 2))
        
        return loss_contrastive

# Reference: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch


# In[13]:


# SIAMESE NETWORK
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.2),  

            nn.Conv2d(64, 64, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.2),

            nn.Conv2d(64, 32, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(32),
            nn.Dropout(p = 0.2),
        )
    
        self.fc1 = nn.Sequential(
            nn.Linear(32*1*16*16, 64),
            nn.ReLU(inplace = True),

            nn.Linear(64, 64),
            nn.ReLU(inplace = True),

            nn.Linear(64, 2))

    def forward_once(self, input):
        output = self.cnn1(input)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Reference: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
# Reference: https://www.kaggle.com/jiangstein/a-very-simple-siamese-network-in-pytorch


# In[14]:


# FUNCTION TO TRAIN THE MODEL
def train_model(model, num_epochs, lr):
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    counter = [] 
    loss_history = []
    iteration = 0 
    num_epochs = num_epochs

    for i, epoch in enumerate(range(num_epochs)):
        print('Currently training epoch {} of {}'.format(epoch + 1, num_epochs))
        model.train()
        epoch_loss = 0
        number_batches = 0

        for j, data in enumerate(train_loader, 0):
            optimizer.zero_grad()  

            seq0, seq1, label = data

            output1, output2 = model(seq0, seq1)

            loss = criterion(output1, output2, label)
            loss.backward()

            optimizer.step() 

            epoch_loss += loss.item()
            number_batches += 1

            if j % 10 == 0:
                print("Epoch number {}".format(epoch + 1))                          
                print("Current loss {}\n".format(loss.item()))
                
        mean_loss = epoch_loss / number_batches
        loss_history.append(mean_loss)
        counter.append(i)
    
    return counter, loss_history

# Reference: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
# Reference: https://www.kaggle.com/jiangstein/a-very-simple-siamese-network-in-pytorch


# # Train the model

# In[15]:


# TRAIN THE MODEL
model = SiameseNetwork()
counter, loss_history = train_model(model, 150, 0.0001)


# # Evaluate the model on the validation set
# 

# In[17]:


# FUNCTION TO MAKE THE DISTANCE MATRIX
def make_distmatrix(model, input0, input1):
    N = len(input0)
    dist_matrix = torch.zeros((N,N))

    output0, output1 = model(input0, input1)
    output0 = output0.cpu()
    output1 = output1.cpu()
    
    for i in range(N):
        distance = F.pairwise_distance(output0[i], output1)
        dist_matrix[i,:] = distance
        
        ind = np.diag_indices(dist_matrix.shape[0])
        dist_matrix[ind[0], ind[1]] = torch.zeros(dist_matrix.shape[0])
        
    return dist_matrix


# In[18]:


# MAKE THE DISTANCE MATRIX OF THE VALIDATION SET
# Convert val to tensors
temp_val0 = torch.Tensor(X_val)
temp_val1 = torch.Tensor(X_val)

# Unsqueeze (add a dimension)
val0 = temp_val0.unsqueeze(1)
val1 = temp_val1.unsqueeze(1)

# Start evaluation mode
model.eval()

val_dist_matrix = make_distmatrix(model, val0, val1)
val_answer = val_dist_matrix.detach().numpy()


# In[19]:


# PERFORMANCE FUNCTION
def calculate_performance_numpy(distances_matrix, labels):
    """
    For a given distance matrix and labels of all samples, this function calculates two performance measures:
     - The mean CMC scores for n = [1, 3, 5, 10]
     - A mean accuracy metric. This metric calculates how many of the k samples that belong to the same class are among
       the first k ranked elements.

    For N samples, the arguments to this function are:
    :param distances_matrix: A NumPy array defining a distance matrix of floats of size [N, N].
    :param labels: An array of integers of size N.

    """
    assert distances_matrix.shape[0] == distances_matrix.shape[1], "The distance matrix must be a square matrix"
    assert len(labels) == distances_matrix.shape[0], "The size of the matrix should be equal to number of labels"

    # Create a bool matrix (mask) where all the elements are True, except for the diagonal
    mask = np.logical_not(np.eye(labels.shape[0], dtype = np.bool))

    # Create a bool matrix (label_equal) with value True in the position where the row and column (i, j)
    # belong to the same label, except for i = j
    label_equal = labels[np.newaxis, :] == labels[:, np.newaxis]
    label_equal = np.logical_and(label_equal, mask)

    # Add the maximum distance to the diagonal
    distances_matrix = distances_matrix + np.logical_not(mask) * np.max(distances_matrix.flatten(), axis = -1)

    # Get the sorted indices of the distance matrix for each sample
    sorted_indices = np.argsort(distances_matrix, axis = 1)

    # Get a bool matrix where the bool values in label_equal are sorted according to sorted_indices
    sorted_equal_labels_all = np.zeros(label_equal.shape, dtype = bool)
    for i, ri in enumerate(sorted_indices):
        sorted_equal_labels_all[i] = label_equal[i][ri]

    # Calculate the mean CMC scores for k=[1, 3, 5, 10] over all samples
    # The score is 1 if a sample j with the same label as i is in the first k ranked positions. It is 0 otherwise
    cmc_scores = np.zeros([4])
    for sorted_equal_labels in sorted_equal_labels_all:
        # CMC scores for a sample
        score = np.asarray([np.sum(sorted_equal_labels[:n]) > 0 for n in [1, 3, 5, 10]])
        # Update running average
        cmc_scores = cmc_scores + score
    cmc_scores /= len(sorted_equal_labels_all)

    # Calculate the accuracy metric

    # Calculate how many samples are there with the same label as any sample i
    num_positives = np.sum(label_equal, axis = 1, dtype = np.int)
    num_samples = len(sorted_equal_labels_all)

    # Calculate the average metric by adding up how many labels correspond to sample i in the first n elements of the
    # ranked row. So, if all the first n elements belong to the same labels the sum is n (perfect score)
    acc = 0
    for i, n in enumerate(num_positives):
        
        acc = acc + np.sum(sorted_equal_labels_all[i, :n], dtype = np.float32) / (n * num_samples)

    return cmc_scores, acc

# Reference: performance function uploaded on Canvas


# In[20]:


# CALCULATE PERFORMANCE VALIDATION SET
print(calculate_performance_numpy(val_answer, y_val))


# # Make a distance matrix of our test set (for answer.txt)

# In[21]:


# CREATE MFCC FEATURES OF THE TEST DATA 
testdata = testdata[0]
testmfcc = []

for n in testdata:
    mfccs_test = librosa.feature.mfcc(n, n_mfcc = 22, sr = 11025)
    testmfcc.append(mfccs_test)


# In[22]:


# CONVERT TO TENSORS
x0 = torch.Tensor(testmfcc)
x1 = torch.Tensor(testmfcc)

# Unsqueeze (add a dimension)
input0 = x0.unsqueeze(1)
input1 = x1.unsqueeze(1)


# In[23]:


# MAKE THE DISTANCE MATRIX OF THE TEST SET
model.eval()
answer = make_distmatrix(model, input0, input1)

# Save as .txt file
answer = answer.detach().numpy()
np.savetxt('answer.txt', answer, fmt = '%1.3f', delimiter = ';')

