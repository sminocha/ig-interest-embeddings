import torch
import torch.nn as nn
import torchvision.models as models

HIDDEN_LAYER_SIZE = 256
mse_loss = nn.BCELoss(size_average=False)

class PoolingEmbedderAE(nn.Module):
    """Contractive AE adapted from https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py"""
    def __init__(self, in_features):
        super(PoolingEmbedderAE,self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, HIDDEN_LAYER_SIZE, bias = False) # Encoder
		self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, in_features, bias = False) # Decoder

        self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
		embedding = self.relu(self.fc1(x.view(-1, self.in_features)))
		return embedding

	def decoder(self, embedding):
		recons_x = self.sigmoid(self.fc2(embedding))
		return recons_x

	def forward(self, x):
            embedding = self.encoder(x)
            recons_x = self.decoder(embedding)
            return embedding, recons_x


def loss_fn(W, x, recons_x, h, lambda):
    mse = mse_loss(recons_x, x)
    dh = h * (1 - h)
    w_sum = torch.sum(W**2, dim=1)
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lamb)


def train(AE, data, batch_size=32, num_epochs=20, lamb=1e-4):
    optimizer = optim.Adam(AE.parameters(), lr = 0.0001)
    N, D = data.shape
    data = transforms.functional.to_tensor(data)
    for epoch in range(num_epochs):
        batch_data = data[iter:iter+batch_size]
        print("Beginning epoch: {}".format(epoch+1))
        iter = 0
        while iter < N:
            optimizer.zero_grad()
            embeddings, recons_x = AE(batch_data)

            # Get the weights
            # AE.state_dict().keys()
            # change the key by seeing the keys manually.
            # (In future I will try to make it automatic)
            W = AE.state_dict()['fc1.weight']
            loss = loss_fn(W, data.view(-1, N), recons_x, embeddings, lamb)
            loss.backward()
            optimizer.step()

            if iter % 100:
                print("\nIteration: {} \nLoss: {}".format(iter, loss.data[0]))

            iter += batch_size


def generate_embeddings(AE, data):
    """Generate pooled embedding using trained autoencoder"""
    N, D = data.shape
    embeddings = np.zeros(N, HIDDEN_LAYER_SIZE)
    for i in range(N):
        embedding = AE.encoder(transforms.functional.to_tensor(data[i]))
        embeddings[i] = embedding.numpy()
    return embeddings


def main():
    # Load embeddings from different sources
    # Concatenate embeddings
    # Instantiate model
    # Train model or load model weights
    # Generate embeddings
    # Save embeddings
    pass
