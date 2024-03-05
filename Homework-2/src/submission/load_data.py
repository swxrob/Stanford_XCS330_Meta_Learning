import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio


'''
RJR: Section for assisting with Debugging
'''
import os
if 'IS_GRADESCOPE_ENV' not in os.environ:
    PRINT_DEBUG = False
else:
    PRINT_DEBUG = False


def get_images(paths, labels, nb_samples=None):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    image_labels = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]

    return image_labels


class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        batch_type,
        config={},
        cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_type: train/val/test
            config: data_folder - folder where the data is located
                    img_size - size of the input images
            cache: whether to cache the images loaded
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train : num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val :]
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        elif batch_type == "val":
            self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

        self.sample_fn = random.sample

        self.shuffle_fn = np.random.shuffle

    def image_file_to_array(self, filename, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / image.max()
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image


    def _sample(self, sample_fn=None, shuffle_fn=None):
        """
        Samples a batch for training, validation, or testing
        Args:
            sample_fn - pointer to a sampling function to be used
            shuffle_fn - pointer to a shuffling function to be used
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and is a numpy array
                2. label batch has shape [K+1, N, N] and is a numpy array
            where K is the number of "shots", N is number of classes
        Note:
            1. The numpy functions np.random.shuffle and np.eye (for creating)
            one-hot vectors would be useful.

            2. For shuffling, remember to make sure images and labels are shuffled
            in the same order, otherwise the one-to-one mapping between images
            and labels may get messed up. Hint: there is a clever way to use
            np.random.shuffle here.

            3. The value for `self.num_samples_per_class` will be set to K+1
            since for K-shot classification you need to sample K supports and
            1 query.

            4. PyTorch uses float32 as default for representing model parameters.
            You would need to return numpy arrays with the same datatype
        """

        if sample_fn is None:
            sample_fn = self.sample_fn

        if shuffle_fn is None:
            shuffle_fn = self.shuffle_fn

        #############################
        ### START CODE HERE ###
        my_name = '_sample'
        if PRINT_DEBUG: print(my_name + ': Starting with: sample_fn=', sample_fn, ', shuffle_fn=', shuffle_fn)

        '''
        Overall Steps:

        (a) Sample N different characters from either the specified train, test, or validation folder.
        (b) Load K + 1 images per character and collect the associated labels, using K images per class for the
            support set and 1 image per class for the query set.
        (c) Format the data and return two tensors, one of flattened images with shape [K + 1, N, 784] and one of
            one-hot labels [K + 1, N, N].

        Notes:
        Note-0: Note that your code only needs to return one single (image, label) tuple. We batch the inputs using an 
        instance of torch.utils.data.DataLoader, and the final shape input images is [B, K + 1, N, 784], 
        and that of the input labels is [B, K + 1, N, N], where B is the batch size.

        Note-a:
        We will sample both the support and query sets as a single batch, hence one batch element should obtain
        image and label tensors of shapes [K + 1, N, 784] and [K + 1, N, N] respectively. 

        Note-b:
        We must shuffle the order of examples in the Query set...
        You should feed the support set examples in the same, fixed order... [No Shuffling!]

        Note-last: 
        We provide helper functions to:
            (1) take a list of folders and provide paths to image files/labels, and 
            (2) to take an image file path and return a flattened numpy matrix. 
        The functions np.random.shuffle and np.eye will also be helpful. Be careful about output shapes and data types!
        '''

        '''
        (a) Sample N different characters from either the specified train, test, or validation folder
            Train / Test / Validation was specified in __init__ with "batch_type" and thus already handled.

            N: self.num_classes
            K+1: self.num_samples_per_class
        '''
        # One hot encoding as Identity matrix NxN (Num character classes X Num character classes)
        labels = np.eye(self.num_classes)

        # Get List of (label, image_path) tuples
        # Useful help and intuition via class Slack:
        #     https://xcs330-scpd.slack.com/archives/C06E6GZBS1Y/p1707187382753819
        #     https://xcs330-scpd.slack.com/archives/C06E6GZBS1Y/p1707100082735929
        #     https://xcs330-scpd.slack.com/archives/C06E6GZBS1Y/p1706371191759319
        # E.g.
        #     "The total number of character types from which you can chose are in self.folders.
        #      As in each folder contains same in a specific writing style, you need to chose your num_classes (N) folders, using sample_fn
        #      You should chose the folder of the class based on the sample_fn"
        language_folders_to_sample = sample_fn(self.folders, self.num_classes)
        label_image_path_pairs = get_images(paths=language_folders_to_sample, labels=labels,
                                            nb_samples=self.num_samples_per_class)

        '''
        (b) Load K + 1 images per character and collect the associated labels
        '''
        label_batch = []
        image_batch = []
        image_fn_batch = []
        for label, image_fn in label_image_path_pairs:
            image_batch.append(self.image_file_to_array(image_fn, dim_input=self.dim_input))
            label_batch.append(label)
            image_fn_batch.append(image_fn)

        '''
        (c) Format the data and return two tensors
            A tuple of (1) Image batch and (2) Label batch:
            - image batch has shape [K+1, N, 784] and is a numpy array
            - label batch has shape [K+1, N, N] and is a numpy array
        '''

        # Note: PyTorch uses float32 as default for representing model parameters.
        #        You would need to return numpy arrays with the same datatype
        # Reshape our image and label batch:
        Kplus1 = self.num_samples_per_class
        N = self.num_classes

        # List => Numpy Arrays shaped appropriately:
        # Test dimensions:
        #     Image batch List[]: 4 entries with 784 items each => need to => np of [K+1, N, 784]
        #     Label batch List[]: 4 entries with   2 items each => need to => np of [K=1, N,   N]
        # TODO: relying on "order='F'" (as in FORTRAN) is an embarassing hack but reshape isn't very flexible. Look for a different way if there's time before final submission.
        image_batch = np.reshape(np.array(image_batch, dtype='float32'), [Kplus1, N, self.dim_input], order='F')
        label_batch = np.reshape(np.array(label_batch, dtype='float32'), [Kplus1, N, N], order='F')
        image_fn_batch = np.reshape(np.array(image_fn_batch), [Kplus1, N, 1], order='F')

        '''
        Shuffle the Query set (not the Support set), using the shuffle_fn provided.
        Note: K+1 = self.num_samples_per_class, i.e. K Support image Sets and 1 Query image Set
        '''
        # For shuffling, remember to make sure images and labels are shuffled in the same order, otherwise the
        # one-to-one mapping between images and labels may get messed up.
        # Hint: there is a clever way to use np.random.shuffle here.
        #
        # From the docs: https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
        #     Multi-dimensional arrays are only shuffled along the first axis

        # Shuffle Query set of Images and Labels in our batch (do not shuffle Support set)
        idx_shuffle = np.arange(len(image_batch[-1,:,:]))
        shuffle_fn(idx_shuffle)
        image_batch[-1,:,:] = image_batch[-1,:,:][idx_shuffle]
        label_batch[-1,:,:] = label_batch[-1,:,:][idx_shuffle]

        return (image_batch, label_batch)

        ### END CODE HERE ###


    def __iter__(self):
        while True:
            yield self._sample()
