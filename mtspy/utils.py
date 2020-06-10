import tarfile
import urllib.request
import os
from scipy.io import mmread
from scipy.sparse import csr_matrix


def get_matrix(Name: str, verbose: bool = True) -> csr_matrix:
    """
    Get matrix from the SuiteSparse Matrix Collection website and
    convert to the scipy.csr format.

    """
    base_url = "https://suitesparse-collection-website.herokuapp.com/MM/"
    url = base_url + Name + ".tar.gz"
    infile = Name.split("/")[1]
    dest_file = infile + '/' + infile + ".mtx"

    # Download the file if it does not exist
    if os.path.isfile(dest_file):
        if verbose:
            print('\t -----------------------------------------------------------')
            print('\t File already exists.')
    else:
        if verbose:
            print('\t -----------------------------------------------------------')
            print('\t Downloading matrix file from suitesparse collection')
        urllib.request.urlretrieve(url, infile + '.tar.gz')

        if verbose:
            print('\t -----------------------------------------------------------')
            print('\t Extrating tar.gz file to folder ./', infile)
        tar = tarfile.open(infile + '.tar.gz')
        tar.extractall()
        tar.close()

    if verbose:
        print('\t -----------------------------------------------------------')
        print('\t Reading matrix and converting to csr format')
    A = mmread(dest_file)
    A = A.tocsr()

    if verbose:
        print('\t -----------------------------------------------------------')
        print("\t Done! \n")

    return A
