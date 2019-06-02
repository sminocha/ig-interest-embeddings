import urllib.request
from bs4 import BeautifulSoup
from skimage import io
import hashlib


def extract_img_url(post_url):
    """From the source of the ig post, extract the url corresponding to the
    first image in the post

    Args:
        post_url (str): ig url string

    Returns:
        img_url (str): image url string
    """
    try:
        ig_post_content = urllib.request.urlopen(post_url).read()
    except:
        print("Bad url: ", post_url)
        return "INVALID_URL"
    page_source = BeautifulSoup(ig_post_content)
    # Get meta tag containing image url
    meta_tag = page_source.find("meta", {"property": "og:image"})
    img_url = meta_tag.get('content')

    return img_url


def download_img(img_url, download_path_base, transforms=[]):
    """Download image from img_url and save to download_path

    Args:
        img_url (str): image url string
        download_path_base (str): first section of path to be used to construct
        absolute path to where image will be saved
        transforms (list): list of functions that transform image before saving.
            These tranforms may include resizing / downsampling functions to
            make sure that all saved images are the same size and resolution.
    """
    # in case we received bad url...
    if img_url == "INVALID_URL":
        return
    download_path = download_path_base + str(abs(hash(img_url)) % (10 ** 8)) + ".png"
    img = io.imread(img_url)
    # Process img
    if transforms:
        for transform in transforms:
            img = transform(img)
    # Save image to download_path
    io.imsave(download_path, img)

def create_path(post_url, download_path, transforms=[]):
    """Given source url of post, put a copy of first image from that post in
    download path.

    Args:
        post_url (str): url of ig post
        download_path (str): absolute path to where image will be saved to
        transforms (list): list of functions that transform image before saving.
            These tranforms may include resizing / downsampling functions to
            make sure that all saved images are the same size and resolution.
    """
    img_url = extract_img_url(post_url)
    download_img(img_url, download_path)
    return download_path


#
# download_path_base = "backbones/ig_downloaded_imgs/"
# transforms = []
# download_img(extract_img_url("https://www.instagram.com/p/BTdRaquBZTD/?taken-by=1misssmeis"), download_path_base, transforms)
