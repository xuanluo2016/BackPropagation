package Implementation;

public class TestBPBipolar {
	public static void main (String arg[]) throws Exception{
		int numInputs = 2;
		int numHidden = 4;
	    int numEpochs = 0;

	    int argA = 0;
	    int argB = 1;
	    
	    double momentumTerm = 0.0;
	    double learningRate = 0.2;
	    
	    double inputs[][] = {{1, 0}, {1, 1},{0, 0}, {0, 1}};
	    double lables[] = {1, -1, -1, 1};

	    double inputVector[]={0, 1};
	    double argValue = 1;
	    double eps = 0.05;	    		
	    // initialize error to any value that is larger than eps
	    double error = eps + 1; 
	   	    
	    // crate a new instance of BackPropagation algorithm
	    BP bpBipolar = new BP(numInputs,numHidden,learningRate,momentumTerm,argA,argB);
		
		// initialize weights
		bpBipolar.initializeWeights();
	    
	    while(error > eps) {
	    	    error = 0;
	    			
	    	// go over epochs to train BP
	    	for (int j=0; j < numHidden; j++) {
	    		inputVector = inputs[j];
	    		argValue = lables[j];
	    		
				// Training BP algorithm with bipolar representation 
			    error = error + bpBipolar.train(inputVector,argValue);		
	    	}
	    	
	    	numEpochs++;

		    System.out.print(numEpochs);
		    System.out.print(" ");
		    System.out.println(error);

			
	    } 

	}
}
