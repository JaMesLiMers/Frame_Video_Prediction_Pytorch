from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import joblib
from torchvision import transforms
from MakeDataset import make_data


class KTHDataset(data.Dataset):
    """`KTHDataset <http://www.nada.kth.se/cvap/actions/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            ``processed/validate.pt`` and  ``processed/test.pt`` exist.
        train (bool or string, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``, If string in 'test/train/validate' then create 
            according dataset.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = [
        'http://www.nada.kth.se/cvap/actions/boxing.zip',
        'http://www.nada.kth.se/cvap/actions/handclapping.zip',
        'http://www.nada.kth.se/cvap/actions/handwaving.zip',
        'http://www.nada.kth.se/cvap/actions/jogging.zip',
        'http://www.nada.kth.se/cvap/actions/running.zip',
        'http://www.nada.kth.se/cvap/actions/walking.zip'
    ]
    sequence_url = 'http://www.nada.kth.se/cvap/actions/00sequences.txt'
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'train.pkl'
    test_file = 'test.pkl'
    validate_file = 'validate.pkl'
    sequence_name = '00sequences.txt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.data = joblib.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            self.data += joblib.load(
                os.path.join(self.root, self.processed_folder, self.validate_file))
            self.data += joblib.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
        elif not self.train:
            self.data = joblib.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
        elif self.train is 'train':
            self.data = joblib.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        elif self.train is 'test':
            self.data = joblib.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
        elif self.train is 'validate':
            self.data = joblib.load(
                os.path.join(self.root, self.processed_folder, self.validate_file))
        else:
            raise NotImplementedError("invalied string input for train, need 'train','test' or 'validate'")
        
        self.test_data()

    def test_data(self):
        for i in self.data:
            sequence = i['sequence']
            for j in range(len(sequence)):
                if j%2 == 0:
                    if sequence[j+1] - sequence[j] <= 20:
                        print(i["filename"] + 'error')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        sequence = self.data[index]["sequence"]
        choice = np.random.randint(low=0, high=len(sequence)//2)*2
        frames = np.random.randint(low=sequence[choice]-1, high=sequence[choice+1] - 20 -1)
        train_frames = self.data[index]["frames"][frames:frames+10]
        gt_frames = self.data[index]["frames"][frames+10:frames+20]
        return train_frames, gt_frames

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.validate_file))


    def download(self):
        """Download the KTH data if it doesn't exist in processed_folder already."""  
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        
        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_dir = os.path.join(self.root, self.raw_folder)
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

            try:
                os.makedirs(file_path.replace('.zip', ''))
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise

            with zipfile.ZipFile(file_path) as zip_f:
                for fileM in zip_f.namelist(): 
                    zip_f.extract(fileM, file_path.replace('.zip', ''))
            os.unlink(file_path)
        
        print('downloading sequence file...')
        data = urllib.request.urlopen(self.sequence_url)
        sequence_name = url.rpartition('/')[2]
        file_dir = os.path.join(self.root, self.raw_folder)
        file_path = os.path.join(self.root, self.raw_folder, sequence_name)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        # process and save as torch files
        print('Processing...')
        train_data, test_data, validate_data = make_data(self.root, self.raw_folder, self.processed_folder, self.sequence_name)
        print(len(train_data))
        print(len(test_data))
        print(len(train_data))

        # Dump data to file.
        print('Dumping train data to file...')
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            joblib.dump(train_data, f)
        print('Dumping test data to file...')
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            joblib.dump(test_data, f)
        print('Dumping validate data to file...')
        with open(os.path.join(self.root, self.processed_folder, self.validate_file), 'wb') as f:
            joblib.dump(validate_data, f)
        
        print('Done!')

if __name__ == "__main__":
    a = KTHDataset('./data/KTHDataset', download=True)
