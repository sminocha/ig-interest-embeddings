import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage import io
import pandas as pd
import timeit
import pickle

EMBEDDING_FILENAME = "cnn_embeddings_2"

def get_feature_extractor(name):
    """Defines and loads pre-trained image feature extractor based on passed
    in name.
    """
    extractor = None
    if name == 'resnet50':
        extractor = models.resnet50(pretrained=True)
        extractor = nn.Sequential(*list(extractor.children())[:-1])
    elif name == 'mobilenet':
        extractor = models.mobilenet_v2(pretrained=True)
        extractor = extractor.features
    if extractor is not None:
        extractor.eval()
    return extractor

def generate_embedding(extractor, image_batch_tensor):
    """Generate numpy image embedding from a batch of images"""
    N, _, _, _ = list(image_batch_tensor.shape)
    img_features = extractor(image_batch_tensor)
    # Flatten features
    img_features = img_features.reshape(N, -1)
    # Merge features for each image into a single embedding vector
    embedding = torch.mean(img_features, dim=0)
    return embedding.detach().numpy()

def create_image_batch(image_paths):
    """Create tensor batch of images from list of image paths"""
    # Load all images from image_paths
    images = []
    for path in image_paths:
        try:
            images.append(io.imread(path))
        except:
            print("Issue loading {}".format(path))
            continue

    processed_images = []
    for image in images:
        pil_image = transforms.functional.to_pil_image(image)
        # resized_image = transforms.functional.resize(pil_image, (224, 224))  # images should already be 224x224
        tensor_image = transforms.functional.to_tensor(pil_image)
        normalized_image = transforms.functional.normalize(tensor_image, mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
        processed_images.append(normalized_image)

    image_batch = torch.stack(processed_images)
    return image_batch


def main():
    # Load csv containing paths to images for each user
    data_path = 'data/image_dataset.csv'
    df = pd.read_csv(data_path, header=0)
    usernames = df['alias'].unique().tolist()
    embeddings_dict = dict()
    # Define extactor
    extractor = get_feature_extractor('resnet50')
    for idx, username in enumerate(usernames):
        if idx < 161:
            continue
        print("Processing user: {}, {}".format(idx, username))
        row = df.loc[df['alias'] == username]
        image_paths = row['downloaded_image'].to_list()[0]
        # Separate image paths
        image_paths = image_paths.split(' ')
        # Create image batch
        image_batch = create_image_batch(image_paths)
        # print(image_batch.shape)
        # Generate embedding for user from batch of images
        start = timeit.default_timer()
        embedding = generate_embedding(extractor, image_batch)
        elapsed = timeit.default_timer() - start
        # print("Generating embedding took {}s".format(elapsed))
        # print(embedding.shape)
        # Store embedding
        embeddings_dict[username] = embedding
        # Save embeddings every 80 users
        if idx % 80 == 0:
            with open(EMBEDDING_FILENAME+'.pkl', 'wb') as f:
                    pickle.dump(embeddings_dict, f)
    # Save final embeddings
    with open(EMBEDDING_FILENAME+'.pkl', 'wb') as f:
            pickle.dump(embeddings_dict, f)

if __name__ == '__main__':
    main()
