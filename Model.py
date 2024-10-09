class Model(nn.Module):
    
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                  embedding_dim,N,kernel,latent_dim,k1,s1,k2,s2,k3,s3,p):
        
        self.N = N
        super(Model, self).__init__()
        
        self.encoder = Encoder(1, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens,
                                k1,s1,k2,s2,k3,s3,p)
        
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1,padding=0)
        
        self.Hyper = Hyper(embedding_dim,N,kernel,latent_dim)

        self.decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens,
                                k1,s1,k2,s2,k3,s3,p)
        
    def forward(self, x):

        batch_size = x.size(0)
        x = x.view(batch_size * 20, 1, 64, 64)

        # Every x is encoded
        z = self.encoder(x)
        z_embed_0 = self.pre_vq_conv(z)
        z_embed = z_embed_0.view(batch_size, 20, embedding_dim,latent_dim*1, latent_dim*1)

        # The first N embeddings are decoded
        z_i = z_embed[:, :self.N, :, :, :].contiguous()
        z_i = z_i.view(batch_size*self.N,embedding_dim,latent_dim,latent_dim) 
        x_i = self.decoder(z_i).view(batch_size,self.N,1,64,64)

        # The Hyper function takes care of the following N embeddings
        Z = self.Hyper(z_embed)
        hyper_z = torch.stack(Z).permute(1,0,4,2,3).contiguous()
        z_h = hyper_z.view(batch_size*self.N,embedding_dim,latent_dim,latent_dim)

        # The embeddings computed by Hyper are decoded and concatenated to the others
        x_h = self.decoder(z_h)
        x_h = x_h.view(batch_size,self.N,1,64,64)
        hyper_full = torch.cat((x_i, x_h), dim=1)
            
        x_out = hyper_full.view(batch_size*20, 1,64, 64)
        x_recon = x_out

        internal_loss = 0

        return x_recon,internal_loss,x_i



model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
            embedding_dim,N,kernel,latent_dim,k1,s1,k2,s2,k3,s3,p).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False,weight_decay=weight_decay)

total_params = sum(p.numel() for p in model.parameters())
print(total_params)
