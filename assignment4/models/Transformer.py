# Code by Sarah Wiegreffe (saw@gatech.edu)
# Fall 2019

import numpy as np

import torch
from torch import nn
import random
import math

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        '''
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        '''        
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # This should take 1-2 lines.                                                #
        # Initialize the word embeddings before the positional encodings.            #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.embed_word = nn.Embedding(input_size, hidden_dim).to(device)
        self.embed_pos = nn.Embedding(max_length, hidden_dim).to(device)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k).to(device)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v).to(device)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q).to(device)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k).to(device)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v).to(device)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q).to(device)
        
        self.softmax = nn.Softmax(dim=2).to(device)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim).to(device)
        self.norm_mh = nn.LayerNorm(self.hidden_dim).to(device)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        self.ffl = nn.Sequential(
                nn.Linear(hidden_dim, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, hidden_dim),
            ).to(device)
        self.nl = nn.LayerNorm(hidden_dim).to(device)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.fl = nn.Linear(hidden_dim, output_size).to(device)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        '''
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups. 

        :returns: the model outputs. Should be normalized scores of shape (N,1).
        '''
        inputs = inputs.to(self.device)
        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling ClassificationTransformer class methods here.  #
        #############################################################################
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return self.final_layer(self.feedforward_layer((self.multi_head_attention(self.embed(inputs)))))
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
        inputs = inputs.to(self.device)
        embeddings_word = self.embed_word(inputs)
        word_pos = torch.Tensor(range(self.max_length)).to(torch.long)
        word_pos = word_pos.to(self.device)
        embeddings_pos = self.embed_pos(word_pos)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return (embeddings_word + embeddings_pos)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.dim_k)
        scores = self.softmax(scores)
        return torch.matmul(scores, v)

    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        inputs = inputs.to(self.device)
        k1 = self.k1(inputs)
        v1 = self.v1(inputs)
        q1 = self.q1(inputs)
        k2 = self.k2(inputs)
        v2 = self.v2(inputs)
        q2 = self.q2(inputs)
        scores_1 = self.attention(q1, k1, v1)
        scores_2 = self.attention(q2, k2, v2)

        concat = torch.cat((scores_1, scores_2), 2)
        output = self.attention_head_projection(concat)
        output = inputs + output
        output = self.norm_mh(output)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return output
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        inputs = inputs.to(self.device)
        outputs = self.ffl(inputs)
        outputs = self.nl(inputs + outputs)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        inputs = inputs.to(self.device)
        outputs = self.fl(inputs)
                
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True