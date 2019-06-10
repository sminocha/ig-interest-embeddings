import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim
import torchvision.transforms as transforms
from embedding_utils import load_embeddings
import numpy as np
import timeit
import pickle

HIDDEN_LAYER_SIZE = 512
mse_loss = nn.MSELoss(reduction='mean')
EMBEDDING_FILENAME = "pool_embeddings.pkl"

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


def loss_fn(W, x, recons_x, h, lamb):
    # print("X shape: {}".format(x.shape))
    # print("recons shape: {}".format(recons_x.shape))
    mse = mse_loss(recons_x, x)
    # dh = h * (1 - h)
    # w_sum = torch.sum(W**2, dim=1)
    # w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    # contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse #+ contractive_loss.mul_(lamb)


def train(AE, data, batch_size=32, num_epochs=100, lamb=1e-4):
    optimizer = optim.Adam(AE.parameters(), lr = 0.0001)
    N, D = data.shape
    # Shuffle data
    shuffled_idxs = np.arange(N)
    np.random.shuffle(shuffled_idxs)
    shuffled_data = data[shuffled_idxs]
    shuffled_data = transforms.functional.to_tensor(shuffled_data).double()
    shuffled_data = shuffled_data.squeeze(0)
    for epoch in range(num_epochs):
        iter = 0
        print("Beginning epoch: {}".format(epoch+1))
        while iter < N:
            batch_data = shuffled_data[iter:iter+batch_size]
            BN, _ = list(batch_data.shape)
            # print(BN)
            optimizer.zero_grad()
            embeddings, recons_x = AE(batch_data)

            # Get the weights
            # AE.state_dict().keys()
            # change the key by seeing the keys manually.
            # (In future I will try to make it automatic)
            W = AE.state_dict()['fc1.weight']
            loss = loss_fn(W, batch_data.view(BN, -1), recons_x, embeddings, lamb)
            loss.backward()
            optimizer.step()

            iter += batch_size

        print("\nIteration: {} \nLoss: {}".format(iter, loss.item()))

        # Save model
        if epoch % 10 == 0:
            print("SAving model")
            torch.save(AE, "poolae.model")


def generate_embedding(AE, data):
    """Generate pooled embedding using trained autoencoder"""
    D = data.size
    data = data.reshape((1, D))
    embedding = AE.encoder(transforms.functional.to_tensor(data).double().squeeze(0))
    embedding = embedding.detach().numpy()
    return embedding

def normalize_embeddings(data):
    """Remove mean and divide by std"""
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return ((data - mean) / std), mean, std


def main():
    save_model = False
    model_path = "poolae.model"
    # Load embeddings from different sources
    embedding_types = ['node2vec', 'lda', 'cnn']
    embedding_paths = ['node2vec_embeddings.pkl', 'lda_embeddings.pkl', 'cnn_embeddings.pkl']
    # Concatenate embeddings
    usernames, embeddings = load_embeddings(embedding_types, embedding_paths, concatenate=True)
    embeddings = embeddings[0]
    print(embeddings.shape)
    N, D = embeddings.shape
    # Normalize embeddings
    embeddings, mean, std = normalize_embeddings(embeddings)
    # Instantiate model
    if save_model:
        # Instantiate model
        PAE = PoolingEmbedderAE(in_features=D).double()
        # Train model
        train(PAE, embeddings)
        # Save model
        torch.save(PAE, model_path)
    else:
        # Instantiate model
        PAE = torch.load(model_path)
        PAE.double()
        PAE.eval()
    print("Training complete")
    # exit()
    # Generate embeddings
    # Define embedding dict
    embeddings_dict = dict()
    # Iterate through each username, get combined embedding, and generate_embedding
    for idx, username in enumerate(usernames):
        print("Processing user: {}, {}".format(idx, username))
        # Generate embedding
        print(embeddings[idx].shape)
        # start = timeit.default_timer()
        embedding = generate_embedding(PAE, embeddings[idx])
        # elapsed = timeit.default_timer() - start
        # print("Generating embedding took {}s".format(elapsed))
        print(embedding.shape)
        # Store embedding
        embeddings_dict[username] = embedding.reshape((HIDDEN_LAYER_SIZE,))
        # exit()
        # Save embeddings every 80 users
        if idx % 80 == 0:
            with open(EMBEDDING_FILENAME+'.pkl', 'wb') as f:
                    pickle.dump(embeddings_dict, f)
    # Save final embeddings
    with open(EMBEDDING_FILENAME+'.pkl', 'wb') as f:
            pickle.dump(embeddings_dict, f)
    # Save embeddings

if __name__ == '__main__':
    main()
