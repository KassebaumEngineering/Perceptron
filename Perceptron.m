
(* Perceptron.m *)

(* :Title: Single Layer Perceptron Neural Network Class *)

(* :Name: Classes`Perceptron` *)

(* :Author: John A. Kassebaum, January 1998. *)

(* :Summary:
   This class implements a simple perceptron neural network type
   using an object oriented approach in Mathematica.  
 *)
 
(* :Context: Classes`Perceptron` *)

(* :Package Version: 1.0 - $Id: Perceptron.m,v 1.7 1998/02/26 02:34:55 jak Exp $ *)

(* :Mathematica Version: 3.0 *)

(* :Copyright: Copyright 1998, John Kassebaum *)

(* :History:
   $Log: Perceptron.m,v $
   Revision 1.7  1998/02/26 02:34:55  jak
   Made a change so it could find "Classes.m". -jak

   Revision 1.6  1998/02/26 02:30:23  jak
   Fixed a typo, and removed unneeded old definitions. -jak

   Revision 1.5  1998/02/26 02:26:02  jak
   I made the Perceptron class a Package.  -jak

   Revision 1.4  1998/02/25 22:49:39  jak
   Added some output functions for the netwowrk paremeters. -jak

   Revision 1.3  1998/02/25 08:08:47  jak
   Changed a print statement and cleaned up the test file. -jak

   Revision 1.2  1998/02/25 07:47:21  jak
   Changed Tabs to spaces. -jak

   Revision 1.1.1.1  1998/02/25 07:41:26  jak
   Initial import of the fully functional Mathematica Perceptron Class. -jak

 *)
 
(* :Keywords:
   Neural Network, Perceptron, Conjugate Gradient
 *)
 
(* :Warning: This is not a standard package!  It requires "Classes.m" from
   the MathSource repository to be placed in the ObjectOriented Directory
   with the other installed code.
 *)

BeginPackage["Classes`Perceptron`",{"Classes`Classes`","Calculus`Master`","Statistics`Master`"}]

new::usage  = "new[Perceptron, numOfInputs, numOfHiddenUnits, numOfOutputs ] 
            generates a new instance of the Perceptron class with the 
			given architecture."
Print::usage  = "Print[ APerceptronInstance ] prints facts about the Instance."
HiddenWeightsOf::usage  = "HiddenWeightsOf[ APerceptronInstance ] returns the 
            Hidden Weight Matrix for the Perceptron Instance."
HiddenBiasesOf::usage  = "HiddenBiasesOf[ APerceptronInstance ] returns the 
            Hidden Bias Matrix for the Perceptron Instance."
OutputWeightsOf::usage  = "OutputWeightsOf[ APerceptronInstance ] returns the 
            Output Weight Matrix for the Perceptron Instance."
OutputBiasesOf::usage  = "OutputBiasesOf[ APerceptronInstance ] returns the 
            Output Bias Matrix for the Perceptron Instance."
ParametersOf::usage  = "ParametersOf[ APerceptronInstance ] returns a list of 
            Parameter Matrices for the Perceptron Instance, in the order
			of Hidden Weights, Hidden Biases, Output Weights, Output Biases."
N::usage  = "N[ APerceptronInstance, inputSamples ] evaluates the network and returns
            a matix of output samples corresponding to the given input samples. "
Error::usage  = "Error[ APerceptronInstance, inputSamples, DesiredOutputSamples ] 
            returns a single value which is typically the sum of squared errors made 
			by all the network outputs."
HiddenFunctionOf::usage  = "HiddenFunctionOf[ APerceptronInstance ] returns a 
            symbolic representation of the function of the given Perceptron 
			instance's hidden layer."
OutputFunctionOf::usage  = "OutputFunctionOf[ APerceptronInstance ] returns a 
            symbolic representation of the function of the given Perceptron 
			instance's output layer."
ErrorFunctionOf::usage  = "ErrorFunctionOf[ APerceptronInstance ]  returns a 
            symbolic representation of the function of the given Perceptron 
			instance's error function."
PrintGradComponents::usage  = "PrintGradComponents[ APerceptronInstance ] prints the 
            symbolic form of the major componenets of the Perceptron instance's
			error gradient function."
ErrorGradient::usage  = "ErrorGradient[ APerceptronInstance, inputSamples, DesiredOutputSamples ] 
            evaluates the network and returns a list of matrices corresponding to the
			error gradients of each major component of the network architecture."
Train::usage  = "Train[ APerceptronInstance, inputSamples, DesiredOutputSamples ] uses
            the Conjugate Gradient Descent algorithm to iteratively improve the perfomance
			of the Neural Network."

Perceptron::usage = "Perceptron is a single hidden-layer neural network
            which uses Tanh functions in the hidden layer and linear 
			units in the output layer. Use 'new' to create one. 
			Perceptron is a sublass of Object."

Begin["`Private`"]

Class[ 
    Perceptron,
    Object, 
    {   numOfInputs,
        numOfHiddenUnits,
        numOfOutputs,
        hidWtVals, 
        hidBsVals,
        outWtVals,
        outBsVals,
        outputlayer,
        hiddenF,
        outputF,
        errorF,
        HiddenLayerGradient,
        HiddenWeightGradient,
        HiddenBiasGradient,
        OutputWeightGradient,
        OutputBiasGradient,
        SymbolicInput,
        SymbolicOutput,
        SymbolicIOVariables,
        ScalarsToVectorsTable
    },
    {    {  new,  (* numOfInputs, numOfHiddenUnits, numOfOutputs *)
            Function[
                new[super]; 
             
             (* default constants and arguments *)
                numOfInputs = #1;
                numOfHiddenUnits = #2;
                numOfOutputs = #3;
               
             (* Pattern Transformations from constant 0 and 1 to Vector 0 and 1 *)
                ScalarsToVectorsTable = Dispatch[{
                    Dot[vec_,0] :> 0, 
                    Dot[0,vec_] :> 0, 
                    Dot[vec_,-0] :> 0, 
                    Dot[-0,vec_] :> 0, 
                    Dot[vec_,1] :> vec,
                    Dot[1,vec_] :> vec,
                    Dot[vec_,-1] :> -vec,
                    Dot[-1,vec_] :> -vec
                }];
                
             (* Initialize symbolic input/output matrices *)
                SymbolicInput  = Array[ Unique[inVar ]&, {1,numOfInputs} ];
                SymbolicOutput = Array[ Unique[outVar]&, {1,numOfOutputs} ];
                SymbolicIOVariables = Flatten[ {SymbolicInput, SymbolicOutput} ];

             (* Intrinsic Functions *)
                outputlayer = H.outputWeights + outputBiases;
                hiddenF = Tanh[ SymbolicInput.hiddenWeights + hiddenBiases ];
                outputF = outputlayer /. H->hiddenF;
                errorF  = (SymbolicOutput - outputlayer).Transpose[SymbolicOutput - outputlayer];
              
             (* Error Gradient Function Expressions *)
                HiddenLayerGradient =
                    ( D[ errorF, H ] 
                        //. ScalarsToVectorsTable
                        /. Times[num_Integer, outputWeights, expr__] -> Times[ num, Dot[ outputWeights, Transpose[ expr ]]] 
                        /. H -> hiddenF
                    );
                HiddenWeightGradient = 
                    ( D[ hiddenF, hiddenWeights ]
                        //. ScalarsToVectorsTable
                    );
                HiddenBiasGradient = 
                    ( D[ hiddenF, hiddenBiases ]
                        //. ScalarsToVectorsTable
                    );
                OutputWeightGradient = 
                    ( D[ errorF, outputWeights ] 
                        //. ScalarsToVectorsTable
                        /. Times[num_Integer, expr1_, expr2__] -> Times[ num, Dot[ Transpose[expr1], expr2 ]] 
                        /. H -> hiddenF 
                    );
                OutputBiasGradient = 
                    ( D[ errorF, outputBiases ] 
                        //. ScalarsToVectorsTable
                        /. H -> hiddenF 
                    ); 

             (* Initialize to random starting parameter values *)
                hidWtVals = Array[ Random[ Real, {-0.1,0.1} ]&, {numOfInputs, numOfHiddenUnits}];
                hidBsVals = Array[ Random[ Real, {-0.1,0.1} ]&, {1,numOfHiddenUnits}          ];
                outWtVals = Array[ Random[ Real, {-0.1,0.1} ]&, {numOfHiddenUnits,numOfOutputs}];
                outBsVals = Array[ Random[ Real, {-0.1,0.1} ]&, {1,numOfOutputs}              ];
                
           ] 
         },
         { Print, (* no arguments *)
           Function[
                Print["numOfInputs:      ", numOfInputs      ];
                Print["numOfHiddenUnits: ", numOfHiddenUnits ];
                Print["numOfOutputs:     ", numOfOutputs     ];
                Print["hiddenWeights: "];   Print[ MatrixForm[ hidWtVals ]];
                Print["hiddenBiases:  "];   Print[ MatrixForm[ hidBsVals ]];
                Print["outputWeights: "];   Print[ MatrixForm[ outWtVals ]];
                Print["outputBiases:  "];   Print[ MatrixForm[ outBsVals ]];
           ]
         },
		 { HiddenWeightsOf, (* return hidden weight matrix *)
		   hidWtVals&
		 },
		 { HiddenBiasesOf, (* return hidden bias matrix *)
		   hidBsVals&
		 },
		 { OutputWeightsOf, (* return output weight matrix*)
		   outWtVals&
		 },
		 { OutputBiasesOf, (* return output bias matrix *)
		   outBsVals&
		 },
		 { ParametersOf, (* no arguments *)
		   ({ hidWtVals, hidBsVals, outWtVals, outBsVals })&
		 },
         { N,
           Function[ inputSamples,
                Module[ {netf, weightDtable, i},
                
                    (* Initialize Weight Value Dispatch Table *)
                    weightDtable = Dispatch[{   
                        hiddenWeights -> hidWtVals, 
                        hiddenBiases  -> hidBsVals,
                        outputWeights -> outWtVals, 
                        outputBiases  -> outBsVals
                    }];
                    
                    netf = Compile[ 
                        Release[ Flatten[ SymbolicInput ]],
                        Release[ outputF //. ScalarsToVectorsTable /. weightDtable ]
                    ];
                    
                    Return[ 
                        Table[
                            Apply[ netf, inputSamples[[i]] ][[1]],
                            {i, Length[ inputSamples ]}
                        ]
                    ];
                ]
            ]
         },
         { Error,
           Function[ {inputSamples,outputSamples},
                Module[ {errfn, weightDtable, i},
                
                    (* Initialize Weight Value Dispatch Table *)
                    weightDtable = Dispatch[{   
                        hiddenWeights -> hidWtVals, 
                        hiddenBiases  -> hidBsVals,
                        outputWeights -> outWtVals, 
                        outputBiases  -> outBsVals
                    }];
                    errfn = Compile[ 
                        Release[ SymbolicIOVariables ],
                        Release[ errorF /. H->hiddenF //. ScalarsToVectorsTable /. weightDtable ] 
                    ];
                    
                    Return[ 
                        Sum[
                            Apply[ errfn, Flatten[ {inputSamples[[i]], outputSamples[[i]]} ]][[1]],
                            {i, Length[inputSamples]}
                        ][[1,1,1]]
                    ];
                ]
           ]
         },
         { HiddenFunctionOf,
           hiddenF&
         },
         { OutputFunctionOf,
           outputF&
         },
         { ErrorFunctionOf,
           errorF&
         },
         { PrintGradComponents, (* no arguments *)
           Function[
               Module[ {weightDtable},
                    (* Initialize Weight Value Dispatch Table *)
                    weightDtable = Dispatch[{   
                        hiddenWeights -> hidWtVals, 
                        hiddenBiases  -> hidBsVals,
                        outputWeights -> outWtVals, 
                        outputBiases  -> outBsVals
                    }];
                    Print[ "dE/dH  = ", HiddenLayerGradient /. weightDtable ];
                    Print[ "dH/dWh = ", HiddenWeightGradient /. weightDtable ];
                    Print[ "dH/dBh = ", HiddenBiasGradient /. weightDtable ];
                    Print[ "dE/dWy = ", OutputWeightGradient /. weightDtable ];
                    Print[ "dE/dBy = ", OutputBiasGradient /. weightDtable ];
               ]
           ]
         },
         { ErrorGradient,  (* Weight Gradient *)
           Function[ {inputSamples, outputSamples},
               Module[ {gradf, weightDtable, i, j},
                    (* Initialize Weight Value Dispatch Table *)
                    weightDtable = Dispatch[{   
                        hiddenWeights -> hidWtVals, 
                        hiddenBiases  -> hidBsVals,
                        outputWeights -> outWtVals, 
                        outputBiases  -> outBsVals
                    }];
                    
                   (* Gradient Function Table *)
                    gradf = List[ 
                        Compile[
                            Release[ SymbolicIOVariables ],
                            Release[ 
                              Module[ {dEdH, dHdWh},
                                dEdH = (HiddenLayerGradient /. weightDtable)[[1,1]] ;
                                dHdWh = Transpose[ (HiddenWeightGradient  /. weightDtable)[[1]] ][[1]];
                                Return[
                                    Sum[
                                        dEdH[[i,1]] dHdWh,
                                        {i, Length[ dEdH ]}
                                    ]
                                ]
                              ]
                            ]
                        ],
                        Compile[
                            Release[ SymbolicIOVariables ],
                            Release[
                              Module[ {dEdH, dHdWh},
                                dEdH = (HiddenLayerGradient /. weightDtable)[[1,1]];
                                dHdBh = (HiddenBiasGradient  /. weightDtable);
                                Return[
                                    Sum[
                                        dEdH[[i,1]] dHdBh,
                                        {i, Length[ dEdH ]}
                                    ]
                                ]
                              ]
                            ]
                        ],
                        Compile[
                            Release[ SymbolicIOVariables ],
                            Release[ (OutputWeightGradient /. weightDtable)[[1,1]] ]
                        ],
                        Compile[
                            Release[ SymbolicIOVariables ],
                            Release[ (OutputBiasGradient /. weightDtable)[[1,1]] ]
                        ]
                    ];
                    
                  (* Calculate and Return Gradient Sum for all input/output samples *)
                    Return[
                      Sum[ 
                        Table[ 
                            Apply[ gradf[[i]], Flatten[ {inputSamples[[j]], outputSamples[[j]]} ]],
                            {i, Length[gradf] }
                        ],
                        {j, Length[ outputSamples]}
                      ]
                    ];
                ]
           ]
         },
         { Train,  (* Via Conjugate Gradient Descent *)
           Function[ {inputSamples,outputSamples,trainingEpochs},
               Module[
                   { P0, P1, G0, G1, W0, W1, Beta, Rtn, RtnErrTmp, ErrMin, LearningRate, weightDtable, i  },
                   Rtn = List[ Error[self,inputSamples,outputSamples] ];
                   Print[ "Initial Error = ", Rtn[[1]] ];
                   Print[ "Training Epochs and Results:" ];

                (* Initialize Weight Value Dispatch Table *)
                    weightDtable = Dispatch[{   
                        hiddenWeights -> hidWtVals, 
                        hiddenBiases  -> hidBsVals,
                        outputWeights -> outWtVals, 
                        outputBiases  -> outBsVals
                    }];
                
                (* Initialize Gradients and Directions *)
                   G1 = ErrorGradient[ self, inputSamples, outputSamples ];
                   P1 = -G1;
                   W1 = { hidWtVals, hidBsVals, outWtVals, outBsVals };

                   For[ i=1, i<=trainingEpochs, i=i+1,
                     (* Prepare for coming iteration *)
                       G0 =. ; G0 = G1;
                       P0 =. ; P0 = P1;
                       W0 =. ; W0 = W1;
                       
                     (* Choose Learning Rate *)
                       ErrMin = Module[ {f ,ef, dispatchTable},
                            dispatchTable = Dispatch[{  
                                hiddenWeights  -> (W0[[1]] + Z P0[[1]]), 
                                hiddenBiases   -> (W0[[2]] + Z P0[[2]]), 
                                outputWeights  -> (W0[[3]] + Z P0[[3]]), 
                                outputBiases   -> (W0[[4]] + Z P0[[4]])
                            }];

                            ef = Compile[ 
                                Release[ Append[ SymbolicIOVariables, X ] ],
                                Release[
                                    ( errorF 
                                        /.  H-> hiddenF 
                                        //. ScalarsToVectorsTable 
                                        /.  dispatchTable 
                                        /.  Z -> X 
                                    )
                                ]
                            ];
                             
                            f = Function[ x,
                                Sum[
                                    Apply[ 
                                        ef,
                                        Flatten[ {inputSamples[[j]], outputSamples[[j]], x} ]
                                    ][[1,1,1,1]],
                                    {j, Length[outputSamples]}
                                ]
                            ];
                            FindMinimum[ f[ LearningRate ], {LearningRate, 0.1, 0.2}]
                       ];
                       
                     (* Update Weights *)
                       W1 = W0 + LearningRate P0 /.ErrMin[[2]];
                       
                      (* Save Weight Values Found *)
                        hidWtVals = . ; hidWtVals = W1[[1]];
                        hidBsVals = . ; hidBsVals = W1[[2]];
                        outWtVals = . ; outWtVals = W1[[3]];
                        outBsVals = . ; outBsVals = W1[[4]];
                       
                      (* Report Progress and Construct Return List *)       
                       RtnErrTmp = ErrMin[[1]]; 
                       Print[ i, ": ", RtnErrTmp, ", ErrMin = ", ErrMin ];
                       AppendTo[ Rtn,  RtnErrTmp ];
                       
                     (* Find new Gradient *)
                       G1 = Grad[ self, inputSamples, outputSamples ];
                       
                     (* Find Beta by Fletcher-Reeves Formula *)
                       Beta = (Flatten[G1].Flatten[G1]) / (Flatten[G0].Flatten[G0]);
                       
                     (* Calculate New Conjugate Direction *)
                       P1 = -G1 + Beta P0; 
                   ];
                   
                   Return[ Rtn ];
               ];
           ]
         }
    }
]

End[]

Protect[ 
    new, 
	Print,
	HiddenWeightsOf,
	HiddenBiasesOf,
	OutputWeightsOf,
	OutputBiasesOf,
	ParametersOf,
	N,
	Error,
	HiddenFunctionOf,
	OutputFunctionOf,
	ErrorFunctionOf,
	PrintGradComponents,
	ErrorGradient,
	Train
]

EndPackage[]
