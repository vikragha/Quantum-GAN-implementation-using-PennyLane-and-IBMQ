# Quantum-GAN-implementation 

This work uses Pytorch, PennyLane and Qiskit to generate new molecules

Reference paper: https://arxiv.org/abs/1805.11973 

Reference paper: https://arxiv.org/abs/1805.11973

Refer DeCao's repo: https://github.com/nicola-decao/MolGAN for baseline

this repo was created to fix some code issues and implement statitical analysis for circuit design and energy consumption. 

Examples:
# installations
!pip install frechetdist


!pip install kora


import kora.install.rdkit


!pip install rdkit-pypi


!pip -q install Pillow


!pip install torch torchvision



#Bash commands to download the QM9 dataset through the '.sh' file


import os


os.chdir("/home/vikram/Quantum-GAN-implementation-using-PennyLane-and-IBMQ/data/")



%%bash


chmod u+x download_dataset.sh


./download_dataset.sh



!python sparse_molecular_dataset.py



import os


os.getcwd()


os.chdir("/home/vikram/Quantum-GAN-implementation-using-PennyLane-and-IBMQ/")




!python main.py --quantum True --layer 2 --qubits 8 --complexity 'nr'


