import urllib.request
from bs4 import BeautifulSoup
from skimage import io


def extract_img_url(ig_url):
    """From the source of the ig post, extract the url corresponding to the
    first image in the post

    Args:
        ig_url (str): ig url string

    Returns:
        img_url (str): image url string
    """
    ig_post_content = urllib.request.urlopen(ig_url).read()
    page_source = BeautifulSoup(ig_post_content)
    # Get meta tag containing image url
    meta_tag = page_source.find("meta", {"property": "og:image"})
    img_url = meta_tag.get('content')

    return img_url


def download_img(img_url, download_path, transforms=[]):
    """Download image from img_url and save to download_path

    Args:
        img_url (str): image url string
        download_path (str): absolute path to where image will be saved to
        transforms (list): list of functions that transform image before saving.
            These tranforms may include resizing / downsampling functions to
            make sure that all saved images are the same size and resolution.
    """
    img = io.imread(img_url)
    # Process img
    if transforms:
        for transform in transforms:
            img = transform(img)
    # Save image to download_path
    io.imsave(download_path, img)
