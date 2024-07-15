#### functions to load, read, write datasets
import os
import sys
import glob
import logging
import shutil


logger = logging.getLogger('ceciestunepipe.util.derivedata')

## eventually some dataset object that saves everything into nmpy memmaped arrays
## so far just trying to get all spikes from a bout dictionary


