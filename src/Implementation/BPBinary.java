package Implementation;

public class BPBinary extends BP {

	public BPBinary(int argNumInputs, int argNumHidden, double argLearningRate, double argMomentumTerm, double argA,
			double argB) {
		super(argNumInputs, argNumHidden, argLearningRate, argMomentumTerm, argA, argB);
	}
	
	/**
	* @param X The input vector. An array of doubles.
	* @return The value returned by th LUT or NN for this input vector
	*/
	public double outputFor (double[] X){
		// Calculate the output for the hidden nodes
		double sum;
		for(int j =0; j < argNumHidden; j++){    
			sum = weight1[0][j]*bias;  //bias term 
			for (int i = 1; i < argNumInputs+1; i++){
			   sum = sum + weight1[i][j]*X[i-1];
			}
		
		hiddenOutputs[j] = super.customSigmoid(sum); // output
		}
		
		// Calculate the output for the output nodes
		sum = weights2[0]*bias; //bias term
		for (int i =1; i < argNumHidden+1; i++){
			sum = sum + weights2[i] *hiddenOutputs[i-1];	
		}
		
		// update output
		output =  super.customSigmoid(sum);
		
		return output;					
	}
	
	   public double computeDelta(double C, double CHat) {
			   return (C-CHat)*CHat*(1-CHat);
	   }
	   
}
