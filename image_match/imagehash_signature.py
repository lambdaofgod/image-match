from image_match.goldberg import ImageSignature
from PIL import Image
import imagehash


class ImageHashSignature(ImageSignature):

    def __init__(self, hashing_fn=imagehash.whash, hash_size=8, n=9, crop_percentiles=(5, 95), P=None, diagonal_neighbors=True,
                 identical_tolerance=2/255., n_levels=1, fix_ratio=False):
        super().__init__(n, crop_percentiles, P, diagonal_neighbors,
                         identical_tolerance, n_levels, fix_ratio)
        self.hashing_fn = hashing_fn
        self.hash_size = hash_size

    def generate_signature(self, path_or_image, bytestream=False):
        im_array = self.preprocess_image(path_or_image, handle_mpo=self.handle_mpo, bytestream=bytestream)
        img = Image.fromarray((im_array * 255).astype('uint8'))
        return self.hashing_fn(img).hash.astype('int8').reshape(-1)
