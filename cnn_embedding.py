import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage import io

def get_feature_extractor(name):
    """Defines and loads pre-trained image feature extractor based on passed
    in name.
    """
    extractor = None
    if name == 'resnet50':
        extractor = models.resnet50(pretrained=True)
        extractor = nn.Sequential(*list(model.children())[:-2])
    elif name == 'mobilenet':
        extractor = models.mobilenet_v2(pretrained=True)
        extractor = extractor.features
    if extractor is not None:
        extractor.eval()
    return extractor

def generate_embedding(extactor, image_batch_tensor):
    """Generate image embedding from a batch of images"""
    img_features = extractor(image_batch_tensor)
    # Merge features for each image into a single embedding vector
    embedding = torch.mean(img_features, dim=0)
    return embedding.numpy()

def create_image_batch(image_paths):
    """Create tensor batch of images from list of image paths"""
    # Load all images from image_paths
    images = []
    for path in image_paths:
        images.append(io.imread(path))

    processed_images = []
    for image in images:
        pil_image = transforms.functional.to_pil_image(image)
        resized_image = transforms.functional.resize(pil_image, (224, 224))
        normalized_image = transforms.functional.normalize(resized_image, mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
        processed_images.append(normalized_image)

    image_batch = torch.stack([transforms.functional.to_tensor(processed_image) in processed_images])
    return image_batch


def main():
    # Load csv containing paths to images for each user
    # For each user/row create tensor batch of images
    # Generate embedding for user from batch of images
    # Store embedding
    # Merge username and embedding
    # Save embeddings
    pass
