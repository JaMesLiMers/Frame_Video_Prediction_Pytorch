from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import sys
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import joblib
from torchvision import transforms


parent_path = os.path.dirname(os.path.dirname(os.getcwd()))
if 'KTHDataset' in os.getcwd():
    os.chdir(parent_path)
sys.path.insert(0, os.getcwd())


from dataloader.KTHDataset.MakeDataset import make_data


class KTHDataset(data.Dataset):
    """`KTHDataset <http://www.nada.kth.se/cvap/actions/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            ``processed/validate.pt`` and  ``processed/test.pt`` exist.
        train (bool or string, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``, If string in 'test/train/validate' then create 
            according dataset.
        data_length (int or None): number of data per epoch
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

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, data_length=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.data_length = data_length

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
        
        # self.test_data()

        # init index list
        # index_list = []
        # while len(index_list) <= self.__len__():
        #     index_list += list(range(len(self.data)))
        # self.index_list = index_list[0:self.__len__()]
        # or random
        self.index_list = np.random.randint(low=0, high=len(self.data), size=self.__len__())

    def test_data(self):
        for i in self.data:
            sequence = i['sequence']
            for j in range(len(sequence)):
                if j%2 == 0:
                    if sequence[j+1] - sequence[j] <= 20:
                        print(i["filename"] + 'error')

    def __len__(self):
        if self.data_length is not int:
            length = 0
            for i in self.data:
                length += i['sequence'][-1]
            self.data_length = length
            return self.data_length
        else:
            return self.data_length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        sequence = self.data[self.index_list[index]]["sequence"]
        choice = np.random.randint(low=0, high=len(sequence)//2)*2
        frames = np.random.randint(low=sequence[choice]-1, high=sequence[choice+1] - 20 -1)
        train_frames = self.data[self.index_list[index]]["frames"][frames:frames+10]
        gt_frames = self.data[self.index_list[index]]["frames"][frames+10:frames+20]


        train_frames = [Image.fromarray(train_frames[i], mode='L') for i in range(10)]
        gt_frames = [Image.fromarray(gt_frames[i], mode='L') for i in range(10)]

        if self.transform is not None:
            train_frames = torch.stack([self.transform(train_frames[i]) for i in range(10)])

        if self.target_transform is not None:
            gt_frames = torch.stack([self.target_transform(gt_frames[i]) for i in range(10)])

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
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of data: {}\n'.format(len(self.data))
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if __name__ == "__main__":
    a = KTHDataset('./data/KTHDataset', download=True)
