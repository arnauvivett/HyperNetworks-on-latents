class Hyper(nn.Module):
    def __init__(self, embedding_dim,N,kernel,latent_dim):
        super(Hyper, self).__init__()

        self.kernel = kernel
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.N = N

        self.theta = nn.Parameter(torch.randn(batch_size, embedding_dim * embedding_dim * kernel * kernel * latent_dim * latent_dim))
        
        self.ReLU = nn.ReLU()

    def forward(self, z_embed):

        batch_size = z_embed.size(0)

        # create a loop to compute every z_t+1 given z_t
        Z = []
        for t in range(20-self.N):

            # Take the initial condition for the hyper network and reate rolling window
            z_0 = z_embed[:, t+self.N-1, :, :, :] 
            z_window = z_embed[:, t:t+self.N, :,:, :]
            z_window = z_window.view(batch_size,(self.N)*self.embedding_dim,self.latent_dim,self.latent_dim)
            
            #Use directly the weights instead of producing them using a hypernetwork
            theta = self.theta.view(batch_size,self.embedding_dim,self.embedding_dim,self.kernel,self.kernel,self.latent_dim,self.latent_dim)
            theta = theta.permute(0,5,6,1,2,3,4)
            
            # Unfold the incoming initial condition and apply the hyper network operation
            z_0_unfold = F.unfold(z_0,self.kernel, stride = 1, padding=self.kernel//2)
            z_0_unfold = z_0_unfold.view(batch_size,self.embedding_dim,self.kernel,self.kernel,self.latent_dim,self.latent_dim)
            z_0_unfold = z_0_unfold.permute(0,4,5,1,2,3)
            z_output = torch.einsum('bhwiokl ,bhwikl -> bhwo',theta, z_0_unfold)
            
            Z.append(z_output)

        return Z
