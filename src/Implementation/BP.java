package Implementation;
import Sarb.NeuralNetInterface;

import java.util.*;
import java.io.*;
public class BP implements NeuralNetInterface{
	
	protected int argNumInputs;
	protected int argNumHidden;
	
	protected double argLearningRate;
	protected double argMomentumTerm;
	protected double argA;
	protected double argB;	
	protected double output = 0.0f;
	protected double bias =1;
	protected double delta; 
	
	protected double weight1[][];
	protected double weights2[];
	protected double preWeights1[][];
	protected double preWeights2[];
	protected double hiddenOutputs[];
	protected double hiddenDeltas[];	

	
	public BP (
			int argNumInputs,
			int argNumHidden,
			double argLearningRate,
			double argMomentumTerm,
			double argA,
			double argB) {
		this.argNumInputs = argNumInputs;
		this.argNumHidden = argNumHidden;
		this.argLearningRate = argLearningRate;
		this.argMomentumTerm = argMomentumTerm;
		this.argA = argA;
		this.argB = argB;
		
		weight1 = new double [argNumInputs+1][argNumHidden];
		weights2 = new double[argNumHidden+1];
		preWeights1 = new double[argNumInputs+1][argNumHidden];
		preWeights2 = new double[argNumHidden+1];
		hiddenOutputs = new double[argNumHidden];
		hiddenDeltas = new double[argNumHidden];
	}
	
	/**
	* intialize weights for both hidden layer and output layer with random values within (-0.5 ~ 0.5)
	*/
	public void initializeWeights()
	{		
		Random generator = new Random();
		
		// initialize weights with random numbers within (-0.5 ~ +0.5)
		for (int i = 0; i < argNumInputs+1; i++)
		  {for (int j = 0; j < argNumHidden; j++){
                weight1[i][j] = generator.nextDouble()-0.5;
		  }
		}	
		for (int i = 0; i < argNumHidden+1; i++){
			weights2[i] = generator.nextDouble()-0.5;
		}
	}
	
	/**
	* Return a bipolar sigmoid of the input X
	* @param x The input
	* @return f(x) = 2 / (1+e(-x)) - 1
	*/
	public double sigmoid (double x){		
		return 2/(1+Math.pow(Math.E, -x)) - 1;	
	}
	
	
	/**
	* This method implements a general sigmoid with asymptotes bounded by (a,b)
	* @param x The input
	* @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
	*/
	public double customSigmoid(double x) {
		return (argB - argA ) / (1+Math.pow(Math.E, -x)) - (-argA);
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
		
		hiddenOutputs[j] = sigmoid(sum); // output
		}
		
		// Calculate the output for the output nodes
		sum = weights2[0]*bias; //bias term
		for (int i =1; i < argNumHidden+1; i++){
			sum = sum + weights2[i] *hiddenOutputs[i-1];	
		}
		
		// update output
		output =  sigmoid(sum);
		
		return output;					
	}
	
	
	/**
	* This methods will update the weights online per each pattern.
	* @param X The input vector
	* @param argValue The new value to learn
	*/
   public void weightUpdate(double[] X, double argValue){
		double C = argValue; // the expected output for X
		
		// weights update for the hidden nodes
		delta = computeDelta(C,output);

		for (int i = 0; i< argNumHidden; i++){
			weights2[i+1] = weights2[i+1] + argMomentumTerm*preWeights2[i+1] + argLearningRate * delta * hiddenOutputs[i];
			preWeights2[i+1] =argMomentumTerm*preWeights2[i+1] + argLearningRate*delta*hiddenOutputs[i];
			}
		
		weights2[0] = weights2[0] +argMomentumTerm*preWeights2[0]+ argLearningRate*delta*bias;
		preWeights2[0] = argMomentumTerm*preWeights2[0] + argLearningRate*delta*bias;
		
		// weights update for the input node
		for (int i = 0; i < argNumHidden; i++){
			hiddenDeltas[i] = hiddenOutputs[i] * (1-hiddenOutputs[i]) * delta * weights2[i+1];
			}
				
		for (int i = 1; i<argNumInputs+1;i++){
			for(int j = 0; j<argNumHidden; j++){
				weight1[i][j]=weight1[i][j]+argMomentumTerm*preWeights1[i][j]+argLearningRate*hiddenDeltas[j]*X[i-1];
				preWeights1[i][j] = argMomentumTerm * preWeights1[i][j] +argLearningRate*hiddenDeltas[j]*X[i-1];
			}
		}
		for (int j = 0; j< argNumHidden; j++){
			weight1[0][j] = weight1[0][j]+ argMomentumTerm * preWeights1[0][j]+ argLearningRate*hiddenDeltas[j]*bias; // bias term
		   preWeights1[0][j] = argMomentumTerm * preWeights1[0][j]+ argLearningRate*hiddenDeltas[j]*bias;
		}
		
	}
	
   public double computeDelta(double C, double CHat) {
		  return (C-CHat)*0.5 *(1-CHat)*(1+CHat);
   }
   
	/**
	* This method will tell the NN or the LUT the output
	* value that should be mapped to the given input vector. I.e.
	* the desired correct output value for an input.
	* @param X The input vector
	* @param argValue The new value to learn
	* @return The error in the output for that input vector
	*/
	public double train (double [] X, double argValue){
		double output_true = argValue;
		double error = 0;
		double out;
		
		try{
			error = 0;			
			out = this.outputFor(X);
			error = 0.5*(output_true-out)*(output_true-out);
			this.weightUpdate(X,argValue);
		}catch (Exception e){
			System.out.println("exception in train function");
		}	

		return error;
	}
	/**
	* A method to write either a LUT or weights of an neural net to a file.
	* @param argFile of type File.
	*/
	public void save(File argFile) {
		
	}
	
	/**
	* Loads the LUT or neural net weights from file. The load must of course
	* have knowledge of how the data was written out by the save method.
	* You should raise an error in the case that an attempt is being
	* made to load data into an LUT or neural net whose structure does not match
	* the data in the file. (e.g. wrong number of hidden neurons).
	* @throws IOException
	*/
	 public void load(String argFileName) throws IOException{
	 }
	 
	/**
	* Initialize the weights to 0.
	*/
	public void zeroWeights() {
		// initialize weights for hidden layer
		for (int i = 0; i < argNumInputs+1; i++){
			for (int j = 0; j < argNumHidden; j++){
		         weight1[i][j] = 0;
			}
		}
		
		// initialize weights for output layer
		for (int i = 0; i < argNumHidden+1; i++){
			weights2[i] = 0;
		}
	}
}



