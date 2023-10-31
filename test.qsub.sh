#This is the most basic QSUB file needed for this cluster.
#   Further examples can be found under /share/apps/examples
#   Most software is NOT in your PATH but under /share/apps
#
#   For further info please read http://hpc.cs.ucl.ac.uk
#   For cluster help email cluster-support@cs.ucl.ac.uk
#
#   NOTE hash dollar is a scheduler directive not a comment.


# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec
  
#$ -l h_rt=1000000
#$ -l gpu=True,gpu_type=a6000
#$ -l tmem=40G
#$ -pe gpu 1

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N mnist_az_test
#$ -cwd

#The code you want to run now goes here.

source /share/apps/source_files/python/python-3.9.5.source
export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH
source /share/apps/source_files/cuda/cuda-11.0.source

source /home/seunghki/skim_py39/bin/activate

python3 test.py
