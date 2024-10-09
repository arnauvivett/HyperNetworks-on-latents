
# dataset parameters
batch_size = 10
num_training_updates = 4000

# parameters for the encoder decoder and residual block
num_hiddens = 40
num_residual_hiddens = 30
num_residual_layers = 3
in_channels = 1
out_channels = 1

#hyper network kernel
kernel = 3

# kernel, stride, and padding values for both encoder and decoder
k1,s1 = 4,2
k2,s2 = 4,2
k3,s3 = 3,1
p = 1

#This function computes the dimension of the output of a convolution given by input 
#dimension (V), kernel (k), stride (s) and padding (p) only for the encoder
def dim_e(k,s,p,V):
    return((V+2*p-k)/s +1)

# dimensions of the latent space
V = 64
V1= dim_e(k1,s1,p,V)
V2 = dim_e(k2,s2,p,V1)
V3_e = dim_e(k3,s3,p,V2)

latent_dim = int(dim_e(1,1,0,V3_e))

#embedding dimension
embedding_dim = 8

# learning dynamics parameters
learning_rate = 1e-4
weight_decay = 0.001

# number of latent spaces to compute the hyper network's parameters
N = 10

