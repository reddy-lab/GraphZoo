"""
Data utils functions for downloading and extracting datasets
"""
import os
from pathlib import Path
from zipfile import ZipFile
import requests

url = { 'cora':  "https://github.com/AnoushkaVyas/GraphZoo/releases/download/Datasets/cora.zip",
        'airport': "https://github.com/AnoushkaVyas/GraphZoo/releases/download/Datasets/airport.zip",
        'citeseer': "https://github.com/AnoushkaVyas/GraphZoo/releases/download/Datasets/citeseer.zip",
        'disease_lp': "https://github.com/AnoushkaVyas/GraphZoo/releases/download/Datasets/disease_lp.zip",
        'disease_nc': "https://github.com/AnoushkaVyas/GraphZoo/releases/download/Datasets/disease_nc.zip",
        'ppi': "https://github.com/AnoushkaVyas/GraphZoo/releases/download/Datasets/ppi.zip",
        'pubmed': "https://github.com/AnoushkaVyas/GraphZoo/releases/download/Datasets/pubmed.zip",
        'webkb': "https://github.com/AnoushkaVyas/GraphZoo/releases/download/Datasets/webkb.zip"
}

def get_url(args):
    """
    Gets the URL of the dataset from the dictionary
    """
    if args.dataset not in list(url.keys()):
            raise ValueError('unknown dataset')
    return url[args.dataset] 

def download_and_extract(args):

    """
    GraphZoo Download and Extract

    Input Parameters
    ----------
        'dataset': (None, 'which dataset to use, can be any of [cora, pubmed, airport, disease_nc, disease_lp] (type: str)')
        'datapath': (None, 'path to raw data (type: str)')

    API Input Parameters
    ----------
        args: list of above defined input parameters from `graphzoo.config`
    
    """
    
    if not args.datapath:
        path = os.path.join(os.getcwd()+'/data/')
        
    else:
        path = args.datapath
        
    filename = args.dataset+'.zip'

    fn = os.path.join(path, filename)

    if not (os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path, exist_ok=True)

    if not os.path.isfile(fn):
        print('%s does not exist..downloading..' % fn)
        url = get_url(args)
        f_remote = requests.get(url, stream=True)
        sz = f_remote.headers.get('content-length')
        assert f_remote.status_code == 200, 'fail to open {}'.format(url)
        with open(fn, 'wb') as writer:
            for chunk in f_remote.iter_content(chunk_size=1024*1024):
                writer.write(chunk)
        print('Download finished. Unzipping the file... (%s)' % fn)
        if not os.path.isfile(fn):
            raise ValueError('Download unsuccessful!')

    else:
        print('zip file already exists! Unzipping...')
        
    
    file = ZipFile(fn)
    file.extractall(path=str(path))
    file.close()


    
