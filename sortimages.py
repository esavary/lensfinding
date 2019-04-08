import glob
import numpy as np
from numpy import genfromtxt
import os, shutil



movetopath='E:\\plotfittingsersic\\trichisquare\\'

originalpath='E:\\plotfittingsersic\\'



my_data = genfromtxt('chiselect.csv', delimiter=',')


list_of_names= my_data[:,0]

for name in list_of_names:

    for file in glob.glob('E:\\plotfittingsersic\\*'+str(int(name))+'*'):
        print os.path.basename(file)
        src = originalpath + os.path.basename(file)
        dst = movetopath + os.path.basename(file)
        shutil.move(src, dst)