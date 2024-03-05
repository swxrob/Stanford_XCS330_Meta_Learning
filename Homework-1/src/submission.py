"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
RJR: Section for Debugging
'''
import os
if 'IS_GRADESCOPE_ENV' not in os.environ:
    PRINT_DEBUG = False
else:
    PRINT_DEBUG = False

class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 embedding_sharing=True):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_sharing = embedding_sharing

        if embedding_sharing:
            self.U, self.Q = self.init_shared_user_and_item_embeddings(num_users, num_items, embedding_dim)
        else:
            self.U_reg, self.Q_reg, self.U_fact, self.Q_fact = self.init_separate_user_and_item_embeddings(num_users, num_items, embedding_dim)

        self.A, self.B = self.init_user_and_item_bias(num_users, num_items)
        self.mlp_layers = self.init_mlp_layers(layer_sizes)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        if self.embedding_sharing:
            predictions, score = self.forward_with_embedding_sharing(user_ids, item_ids)
        else:
            predictions, score = self.forward_without_embedding_sharing(user_ids, item_ids)

        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score
    
    def init_shared_user_and_item_embeddings(self, num_users, num_items, embedding_dim):
        """
        Initializes shared user and item embeddings
        used in both factorization and regression tasks

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.
            

        Returns
        -------

        U: ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q: ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        """
        U = Q = None
        ### START CODE HERE ###
        my_name = "init_shared_user_and_item_embeddings"
        if PRINT_DEBUG:
            print(my_name+': STARTING with: num_users=', num_users, ", num_items=", num_items, "embedding_dim=", embedding_dim)

        U = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        Q = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        ### END CODE HERE ###
        return U, Q
    
    def init_separate_user_and_item_embeddings(self, num_users, num_items, embedding_dim):
        """
        Initializes separate user and item embeddings
        where one will be used for factorization (ie _fact) and 
        other for regression tasks (ie _reg)

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.
            

        Returns
        -------

        U_reg: first ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_reg: first ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        U_fact: second ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_fact: second ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)

        Note: Order does matter here! Please declare the layers in the order
        they are returned.
        """
        U_reg = Q_reg = U_fact = Q_fact = None
        ### START CODE HERE ###

        my_name = 'init_separate_user_and_item_embeddings'
        if PRINT_DEBUG: print(my_name+': STARTING with, num_users=', num_users, 'num_items=', num_items, ', embedding_dim=', embedding_dim)

        U_reg = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        Q_reg = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        U_fact = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        Q_fact = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        ### END CODE HERE ###
        return U_reg, Q_reg, U_fact, Q_fact
    
    def init_user_and_item_bias(self, num_users, num_items):
        """
        Initializes user and item bias terms

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.

        Returns
        -------

        A: ZeroEmbedding layer for users
            nn.Embedding of shape (num_users, 1)
        B: ZeroEmbedding layer for items
            nn.Embedding of shape (num_items, 1)
        """
        A = B = None
        ### START CODE HERE ###
        my_name = 'init_user_and_item_bias'
        if PRINT_DEBUG: print(my_name+': STARTING with: num_users=', num_users, 'num_items=', num_items)

        A = ZeroEmbedding(num_embeddings=num_users, embedding_dim=1)
        B = ZeroEmbedding(num_embeddings=num_items, embedding_dim=1)

        ### END CODE HERE ###
        return A, B
    
    def init_mlp_layers(self, layer_sizes):
        """
        Initializes MLP layer for regression task

        Parameters
        ----------

        layer_sizes: list
            List of layer sizes to for the regression network.

        Returns
        -------

        mlp_layers: nn.ModuleList
            MLP network containing Linear and ReLU layers
        """
        mlp_layers = None
        ### START CODE HERE ###
        my_name = 'init_mlp_layers'

        if PRINT_DEBUG: print(my_name+': STARTING with: layer_sizes=', layer_sizes)

        ''' 
        RJR: Notes
        
        Each layer (except the final layer) should be followed by a ReLU activation.
        The final layer should output the final user-item predicted score in and have an output
        size of 1.
        
        Using nn.ModuleList() ensures the gradients pass between the layers because their tensors are co-Registered.
        '''
        mlp_layers = nn.ModuleList()
        n_layers = len(layer_sizes)
        for i in range(n_layers):

            # Layers 1 through N-1: Linear followed by a ReLU activation
            if i < n_layers - 1:
                mlp_layers.append(nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i + 1]))
                mlp_layers.append(nn.ReLU())

            # Last layer: Only Linear, no ReLU, and output final user-item predicted score and output of size 1.
            else:
                mlp_layers.append(nn.Linear(in_features=layer_sizes[i], out_features=1))

        if PRINT_DEBUG: print(my_name+': ENDING with: mlp_layers=', mlp_layers)

        ### END CODE HERE ###
        return mlp_layers

    def forward_with_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None
        ### START CODE HERE ###

        '''
        RJR: Notes
        From the Handout: Both forward methods above receive a batch of (userIDi, itemIDj) of user-item pairs. 
        The model should output a probability pij of shape (batch size,) that user i would watch movie j, given by Eq. 1,
        and a predicted score rˆij of shape (batch size,) the user i would assign to movie j, given by Eq. 4.

        Recall we're using nn.ModuleList() => Need to loop through the layers manually.
                
        Equation 1: Predictions: 
        Interaction Probability User_i and Movie_j:
            pij = ui^T * qj + ai + bj
            where,
            ai : user i specific bias term
            bj : movie j specific bias term
            
        Equation 4: Aka Score:
            rˆij = f_θ([ui, qj, ui ∗ qj ])

        Steps:
        1. Task 1: User-Item Pair Likelihood Score 
        The likelihood Score for a user-item pair is given by the dot product of the item and user latent vectors, i.e. Equation 1.
        a. Using the same latent vector representations as before, we will concatenate [ui, qj , ui ∗qj ] 
         (where ∗ denotes element-wise multiplication) together and pass it through a neural network with a single hidden layer
        
        2. Task 2: Numerical Score, aka Rating
        The numerical score is predicted using a small Multi Layer Perceptron (MLP), i.e. Equation 4.        
        '''
        my_name = 'forward_with_embedding_sharing'

        if PRINT_DEBUG:
            print(my_name+': STARTING with: user_ids.shape=', user_ids.size(), ', item_ids.shape=', item_ids.size())
        '''
        Task 1: User-Item Pair Likelihood Score
                Eq. 1: pij = ui^T * qj + ai + bj
        '''
        # integrate over the embedding dimension
        uiTqj = torch.sum(self.U(user_ids) * self.Q(item_ids), dim=1)
        ai = torch.squeeze(self.A(user_ids), dim=1)
        bj = torch.squeeze(self.B(item_ids), dim=1)
        pij = uiTqj + ai + bj
        predictions = pij
        assert(predictions.size() == torch.Size([user_ids.size(0)]))

        ''' 
        Task 2: Numerical Score, aka Rating using MLP (our Module_List)
                Eq. 4:  rˆij = f_θ([ui, qj, ui ∗ qj ])
        '''
        # Concatenate ui, qi, ui*qi along the embedding dimension (dim=1)
        input_mlp = torch.concat([self.U(user_ids), self.Q(item_ids), self.U(user_ids)*self.Q(item_ids)],dim=1)
        # Check Shape
        assert(input_mlp.size() == torch.Size([user_ids.size(0), 3*self.embedding_dim]))

        # Pass through our MLP network:
        #     We have to step through the network one layer at a time because Module_List doesn't have it's own "forward()" function.
        #     Another idea is to wrap Module_List inside Sequential like this:
        #         https://discuss.pytorch.org/t/notimplementederror-module-modulelist-is-missing-the-required-forward-function/175049/2
        #     TODO: Didn't try it, may try this later.
        rij = input_mlp
        for layer in self.mlp_layers:
            rij = layer(rij)
        rij = torch.squeeze(rij, dim=1)     # remove extra dimension of size 1
        score = rij

        if PRINT_DEBUG:
            print(my_name+': ENDING with: predictions.shape=', predictions.size(), ', and score.shape=', score.size())

        ### END CODE HERE ###
        return predictions, score
    
    def forward_without_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None
        ### START CODE HERE ###

        my_name = 'forward_without_embedding_sharing'
        '''
         RJR: Notes
         From the Handout: Both forward methods above receive a batch of (userIDi, itemIDj) of user-item pairs. 
         The model should output a probability pij of shape (batch size,) that user i would watch movie j, 
         given by Eq. 1 and a predicted score rˆij of shape (batch size,) the user i would assign to movie j, given by Eq. 4.

         Recall we're using nn.ModuleList().

         Equation 1: Predictions: 
         Interaction Probability User_i and Movie_j:
             pij = ui^T * qj + ai + bj
             where,
             ai : user i specific bias term
             bj : movie j specific bias term

         Equation 4: Aka Score:
             rˆij = f_θ([ui, qj, ui ∗ qj ])

         Steps:
         1. Task 1: User-Item Pair Likelihood Score 
         The likelihood Score for a user-item pair is given by the dot product of the item and user latent vectors, i.e. Equation 1.
         a. Using the same latent vector representations as before, we will concatenate [ui, qj , ui ∗qj ] 
          (where ∗ denotes element-wise multiplication) together and pass it through a neural network with a single hidden layer

         2. Task 2: Numerical Score, aka Rating
         The numerical score is predicted using a small Multi Layer Perceptron (MLP), i.e. Equation 4.        
         '''
        my_name = 'forward_with_embedding_sharing'

        if PRINT_DEBUG:
            print(my_name + ': STARTING with: user_ids.shape=', user_ids.size(), ', item_ids.shape=', item_ids.size())
        '''
        Task 1: User-Item Pair Likelihood Score
                Eq. 2: 
                pij+ = ui^T * qj + ai + bj
                pij'- = ui^T * qj' + ai + bj'
        '''
        # integrate over the embedding dimension to ensure we aren't broadcasting ai and bj in the summation later
        uiTqj = torch.sum(self.U_fact(user_ids) * self.Q_fact(item_ids), dim=1)

        ai = torch.squeeze(self.A(user_ids), dim=1)
        bj = torch.squeeze(self.B(item_ids), dim=1)
        pij = uiTqj + ai + bj
        predictions = pij
        assert(predictions.size() == torch.Size([user_ids.size(0)]))

        ''' 
        Task 2: Numerical Score, aka Rating using MLP (our Module_List)
                Eq. 4:  rˆij = f_θ([ui, qj, ui ∗ qj ])
        '''
        # Concatenate ui, qi, ui*qi along the embedding dimension (dim=1)
        input_mlp = torch.concat(
            [self.U_reg(user_ids), self.Q_reg(item_ids), self.U_reg(user_ids) * self.Q_reg(item_ids)], dim=1)
        # Check shape is as expected:
        assert (input_mlp.size() == torch.Size([user_ids.size(0), 3 * self.embedding_dim]))

        # Pass through our MLP network:
        #     We have to step through the network one layer at a time because Module_List doesn't have it's own "forward()" function.
        #     Another idea is to wrap Module_List inside Sequential like this:
        #         https://discuss.pytorch.org/t/notimplementederror-module-modulelist-is-missing-the-required-forward-function/175049/2
        #     TODO: Didn't try it, may try this later.
        rij = input_mlp
        for layer in self.mlp_layers:
            rij = layer(rij)
        rij = torch.squeeze(rij, dim=1)  # remove extra dimension of size 1
        score = rij

        if PRINT_DEBUG:
            print(my_name + ': ENDING with: predictions.shape=', predictions.size(), ', and score.shape=', score.size())

        ### END CODE HERE ###
        return predictions, score