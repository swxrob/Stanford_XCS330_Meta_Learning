import torch
from torch import nn, Tensor
import torch.nn.functional as F

'''
RJR: Section for assisting with Debugging
'''
import os
if 'IS_GRADESCOPE_ENV' not in os.environ:
    PRINT_DEBUG = False
else:
    PRINT_DEBUG = False


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        ### START CODE HERE ###
        '''
        Additional info from the Handout:

        Inputs are an image tensor of shape [B, K + 1, N, 784] and a label tensor of shape [B, K + 1, N, N]
        Output (ground truth) labels of shape [B, K + 1, N, N]. 

        The layers to use have already been defined for you in the init function. 
        Hint: Remember to pass zeros, not the ground truth labels for the final N examples.
        '''
        my_name = 'mann:forward()'

        if PRINT_DEBUG:
            print(my_name+': STARTING with: input_images: ', input_images.shape, ', input_labels: ', input_labels.shape)

        B, Kplus1, N, image_dim = input_images.shape
        if PRINT_DEBUG:
            print(my_name+': Shapes: B, Kplus1, N, image_dim: ', B, Kplus1, N, image_dim)

        '''
        0. Make copies of the Input Image and Label Support and Query Sets, Zero the Query Labels and Concatenate

           To feed N*Kplus1 labeled examples through the network efficiently, we reshape our image and label tensors to be 
           network compatible as in [B, N*Kplus1, 784 or N]
        '''
        # Copies so we don't mess up the originals
        input_images = input_images.clone()
        input_labels = input_labels.clone()

        # Reshape Image Set to [B, N*Kplus1, 784]
        input_images = input_images.reshape(B, N*Kplus1, image_dim)

        # Set the Query set labels to Zeros
        input_labels[:,-1,:,:] = torch.zeros(input_labels[:,-1,:,:].shape)
        # Reshape Label Set to [B, N*Kplus1, N]
        input_labels = input_labels.reshape(B,N*Kplus1,N)

        '''
        1. For each Image, and Class, pass Concatenated Image + Label through Network
        '''
        # Concatenate Batch of Images and Labels
        input_images_and_labels = torch.concatenate([input_images, input_labels], dim=2)

        '''
        2. Pass Support and Query Images and Labels through the Network
        '''
        # TODO: Should we be propogating or initializing the hidden and cell states for our 2 LSTM's? (h0 and c0 default to 0 otherwise)
        out_layer1, _ = self.layer1(input_images_and_labels)
        out_layer2, _ = self.layer2(out_layer1)

        '''
        3. Predicted Labels: Final Network Output
           Unravel these from [B, N*Kplus1, N] to [B, Kplus1, N, N]
        '''
        output = out_layer2.reshape(B, Kplus1, N, N)

        # Check shape of predictions is correct: [B, K+1, N, N]
        assert output.shape == torch.Size([B, Kplus1, N, N]), (
                my_name + ': FAIL: Wrong Network Output shape: ' + str(output.shape))

        return output

        ### END CODE HERE ###

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################

        loss = None

        ### START CODE HERE ###
        '''
        Notes from the Handout:
        - The loss is computed between the query set predictions and the ground truth labels, which is 
          then backpropagated through the network (backpropogation happens in main.py, not here). 
        - The loss is only computed on the set of N query images, which comprise of the last examples 
          from each character class.
        '''

        my_name = 'mann.py:loss_function()'
        if PRINT_DEBUG:
            print(my_name+': STARTING with preds: ', preds.shape, ', labels: ', labels.shape)

        '''
        Return a Single Scalar across the N test (Query) images:
            Loss should only be calculated on the N test images.
            Shapes are:
                preds:  [B, K+1, N, N] network output
                labels: [B, K+1, N, N] labels
        '''

        '''
        From the class Pytorch tutorial, 
            https://colab.research.google.com/drive/1DrVS4Ume6ZYopiFcnzNTnm0LcddgjWgH?usp=share_link#scrollTo=v1MUJ0cfxD58
        "if you're used to using TensorFlow's CategoricalCrossentropy loss,
        note that F.cross_entropy takes integer labels, not one-hot labels"

        From Class Slack: Angela Liu (CF) 15 days ago
        For 2-1, a common issue I’ve seen for the loss function is that you need to make sure the class dim of your 
        inputs is the second dim (dim=1). If you look at the pytorch documentation, for any tensors of 3+ dim, the 
        cross entropy loss will treat the second dimension (dim=1) as the class dimension. (By default in our case, 
        the class dim is the last dim.) So you will either need to reshape your tensor or swap the dimensions so that 
        the dim you are taking cross entropy loss over actually corresponds to the class distribution.
        
        Other ideas:
        https://discuss.pytorch.org/t/categorical-cross-entropy-loss-function-equivalent-in-pytorch/85165
        '''

        '''
        CF assistance:
        Your issue seems to be in the loss_function where you are using the K-dimensional CELoss. 
        Transform the input/predictions to (-1, C) and labels to (-1,) and apply CELoss.
        
        From the PyTorch CrossEntropyLoss documentation:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        input - The input is expected to contain the unnormalized logits for each class 
                (which do not need to be positive or sum to 1, in general). 
                Input has to be a Tensor of size (C) or (minibatch, C)
                
        target - Class indices in the range [0,C), where C is the number of classes.
        '''

        B, Kplus1, N, _ = labels.shape

        # Slice to the N Test/Query images (dim=1), then swap axes for the CCE function:
        # preds_cce  = torch.swapaxes(preds[ :,-1,:,:], 1, 2)
        # labels_cce = torch.swapaxes(labels[:,-1,:,:], 1, 2)

        # Reshape our predicted labels (classes) to [B*N, N]
        preds_cce  = preds[:, -1, :, :].reshape(B*N,N)

        # Reshape our truth labels to [B*N,N] then convert to integers so we have [B*N,] as input
        labels_cce = labels[:,-1,:,:].clone().reshape(B*N,N).argmax(axis=1)

        loss = torch.nn.functional.cross_entropy(input=preds_cce,target=labels_cce)

        # import torchviz
        # torchviz.make_dot(loss).render("graph.png")

        ### END CODE HERE ###

        return loss
