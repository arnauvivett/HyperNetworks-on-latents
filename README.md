# HyperNetworks-on-latents



This is a meta learning architecture based on hypernetworks. The objective is to let the network produce the evolution rules of a sequence of frames instead of directly learning it. 

<img width="741" alt="hyper" src="https://github.com/user-attachments/assets/1ae89a35-be5b-404f-a090-03723f4c2828">

This is done by first encoding the first N frames into a latent space. Those representations are used as input to a primary network, wich produces the weights $\Phi$ of the hypernetwork. The hypernetwork is a spatial wise convolutional neural network that is applied to the last available latent representation in order to get the next one. Here are the results using Moving MNIST:

<img width="575" alt="predictions" src="https://github.com/user-attachments/assets/4a6fe194-00bd-4d00-b930-b0395f2a7402">

The first image is the true data, the second image is separated into the first row, being the autoencoder reconstructions where N=10 and the second row are the hypernetwork reconstructions. The network predicts one frame at a time, and every time it is given the true previous N frames as a sliding window. 

To reproduce the results, follow the order of execution: 
- Imports
- Hyperparameters
- Dataset
- MainNetwork (Autoencoder part, based on the code available at [1])
- Hyper / Flat (If you want to check the difference between learning the hypernetwork and the direct approach)
- Main model
- Training and testing


[1]: https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
