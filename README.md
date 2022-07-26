# Introduction
Here, we have shown how machine learning configuration interaction (MLCI) method works with single step training. 
For this specific study, artificial network (ANN) is the ML part and it is a single-layer feed-forward network. 
The model is built with early steps of Monte Carlo Configuration Interaction (MCCI) and predicts the whole Hilbert 
space. Here it is worthy to point out that Coe and co-workers have developed a MLCI approach with multiple training 
steps along with evolution of the space using singles doubles connected configurations. In this code, we attempt a 
single step training and have an improved model which enables us to predict the important configurations even when 
the space is distant from the training space.

Details of the algorithm can be summarized in three elementary steps:

1. Building model: 		As stated earlier, MCCI method has been used to generate training data for ANN. For MCCI, one and two 
                                electron integrals of Hartree-Fock orbitals is given in FCIDUMP format. Data is taken from the initial 
				steps of MCCI. This has two advantages - (i) only few diagonalization steps are required; (ii) the MLCI
				model thus built is from a balanced data set of both important and unimportant configurations. 
                   		The training data is an accumulation of n-steps. Duplicate configurations are screened and weights are 
 				taken from last step where it has been encountered.
    
2. Prediction with the model: 	After training, the models are then used to predict in the complete Hilbert space. 
				ANN with above mentioned architecture is trained with three output domains. 
    			      	Activation functions are sigmoid for the models where the outputs are abs-CI and transformed CI.
                              	In case of, log CI output values, the model is built with rectified linear Unit (ReLU) activation 
                              	functions. 
    
3. Variational energy 
   calculation:  		The most important M configurations are chosen from these predicted values and used for
    				calculation of variational energies with Davidson diagonalization.

# Prerequisites :
1. Python 3.0+
2. mcci-master
3. Molpro (https://www.molpro.net/)

# Contributors :
1. Madhumita Rano
2. Prof. Debashree Ghosh

# Compilation :
	a) First, download and install the mcci-master programme using instruction therein : git clone https://github.com/MCCI/mcci.git
	b) Then install the variational energy calculator, "varE" and user is adviced to follow same procedure for installation as above.

#How to run this code ?
1. Download "csv_data.zip" from "Releases" and then unzip csv_data.zip
2. Modify input arguments in "general_input.in" file accordingly. 
3. Run 
   ./build_model
   ./assemble

N.B.: Please give your "varE" executable path in "cal_energy.py" for calculating variational energy.

# Input arguments 
1. nalpha        : 	INT
       			Number of alpha electrons (is equal to beta electrons for spin = 0 states)
2. dspace        : 	INT
			Subspace size for variational energy calculation
3. n_inputs	 :	INT
			Number of spin orbitals
4. H		 :	INT
			Number of nodes in the hidden layer.
5. lr		 :	FLOAT
			Learning rate of the optimizer of ANN.
6. epoch	 :	INT
			Number of iterations needed to optimize the model.

User is suggested to look at python code (trainANN.py) to tune parameters for better training. This sample codes demonstrate only a Log-MLCI
with associated variational energy calculation. Other MLCI models like Abs-MLCI and Transformed-MLCI can be easily build with this programme
with change in ANN target output and activation functions. Also, it shows the methodology for a specific geometry, user has to repeat it 
throughout the geometry coordinates to get the complete potential energy surface (PES).



# Files needed (or to be constructed) before
1. input.csv
2. ann_format.csv
3. FCIDUMP
A sample input file for generating FCIDUMP of water molecule is given within "molpro_files". Some other softwares to generate this integral 
file are Q-Chem(http://www.q-chem.com/), HORTON (https://theochem.github.io/horton/), PSI4 (http://psicode.org/).



# Generated output files
After successful running both build_model & assemble, there will following output files. 
1. e_summary (variational energy for this subspace) & civ_out (wave function)
2. Training_loss (Changes in loss function during each step training)
3. validation.out & train.out (training and validation output files)
4. saved_model.pth (saved model with optimized ANN parameters)
5. file (train and validation MSE with sensitivity and specificity values)
6. time_run (time needed for different steps of calculations)
