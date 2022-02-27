The program realization of our paper is based on this paper, The author of the source code is as follows.

a deep learning tool for the classification of biological sequences

autoBioSeqpy is published at Journal of Chemical Information and Modeling. The link is https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00409 and the related BibTeX is provided below:

@article{jing2020autobioseqpy,
  title={autoBioSeqpy: a deep learning tool for the classification of biological sequences},
  author={Jing, Runyu and Li, Yizhou and Xue, Li and Liu, Fengjuan and Li, Menglong and Luo, Jiesi},
  journal={Journal of Chemical Information and Modeling},
  volume={60},
  number={8},
  pages={3755--3764},
  year={2020},
  publisher={ACS Publications}
}



Usage
1. File 'data' contains training and testing data used in this study. Ptrain and Ntrain represent the training set of positive samples and negative samples, respectively
Ptest and Ntest represent the test set of positive samples and negative samples, respectively.
2. The analysisPlot.py is used to output some evaluation graphics.
3. The dataProcess.py processes sequence data into vectors or matrices.
4. The moduleRead.py is used to read the compiled modules from .py file. 
5. The paraParser.py is used to introduce basic information about variables.
6.  CNN-biLSTM-attention_model is the architecture of the neural network model, which can be trained and tested by calling running.py. 

=================================

We use the following commands to select parameters, and realize model training and independent testing at the same time.
###
python running.py --dataType dna --dataEncodingType dict  --dataTrainFilePaths using/Ptrain.txt using/Ntrain.txt --dataTrainLabel 1 0 --dataTestFilePaths using/Ptest.txt using/Ntest.txt --dataTestLabel 1 0 --modelLoadFile CNN-biLSTM-attention_model.py --verbose 1 --outSaveFolderPath output --savePrediction 1 --saveFig 1 --batch_size 80 --epochs 20 --spcLen 300 --useKMer 1 --KMerNum 2 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt                                         


