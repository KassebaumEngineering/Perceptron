
(* Perceptron.m *)

(* :Title: Single Layer Perceptron Neural Network Class *)

(* :Name: Classes`Perceptron` *)

(* :Author: John A. Kassebaum, January 1998. *)

(* :Summary:
   This class implements a simple perceptron neural network type
   using an object oriented approach in Mathematica.  
 *)
 
(* :Context: Classes`Perceptron` *)

(* :Package Version: 1.0 - $Id: Perceptron.m,v 1.10 1998/02/27 04:09:59 jak Exp $ *)

(* :Mathematica Version: 3.0 *)

(* :Copyright: Copyright 1998, John Kassebaum *)

(* :Keywords:
   Neural Network, Perceptron, Conjugate Gradient
 *)
 
(* :Warning: This is not a standard package!  It requires "Classes.m" from
   the MathSource repository to be placed in a Directory called Classes in
   the AddOns/Applications subdirectory of the Mathematica Installation.
 *)

BeginPackage[ "Classes`Perceptron`",
  { "Classes`Classes`",
    "Statistics`Master`",
	"Calculus`Master`"
  }
]

new::usage  = "new[Perceptron, numOfInputs, numOfHiddenUnits, numOfOutputs ] 
            generates a new instance of the Perceptron class with the 
			given architecture."
Print::usage  = "Print[ APerceptronInstance ] prints facts about the Instance."
GetHiddenWeights::usage  = "GetHiddenWeights[ APerceptronInstance ] returns the 
            Hidden Weight Matrix for the Perceptron Instance."
GetHiddenBiases::usage  = "GetHiddenBiases[ APerceptronInstance ] returns the 
            Hidden Bias Matrix for the Perceptron Instance."
GetOutputWeights::usage  = "GetOutputWeights[ APerceptronInstance ] returns the 
            Output Weight Matrix for the Perceptron Instance."
GetOutputBiases::usage  = "GetOutputBiases[ APerceptronInstance ] returns the 
            Output Bias Matrix for the Perceptron Instance."
GetParameters::usage  = "GetParameters[ APerceptronInstance ] returns a list of 
            Parameter Matrices for the Perceptron Instance, in the order
			of Hidden Weights, Hidden Biases, Output Weights, Output Biases."
N::usage  = "N[ APerceptronInstance, inputSamples ] evaluates the network and returns
            a matix of output samples corresponding to the given input samples. "
Error::usage  = "Error[ APerceptronInstance, inputSamples, DesiredOutputSamples ] 
            returns a single value which is typically the sum of squared errors made 
			by all the network outputs."
GetOutputFunction::usage  = "GetOutputFunction[ APerceptronInstance ] returns a 
            compiled function which computes the values of the given Perceptron 
			instance's output layer given a collection of inputs and parameters."
GetErrorFunction::usage  = "GetErrorFunction[ APerceptronInstance ]  returns a 
            compiled function which computes the values of the given Perceptron 
			instance's sum of squared errors given a collection of inputs, outputs,
			and parameters."
GetGradientFunction::usage  = "GetGradientFunction[ APerceptronInstance ]  returns a 
            compiled function which computes the values of the given Perceptron 
			instance's error gradient function given a collection of inputs, outputs,
		    and parameters."
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
		Params,
        OutputFn,
        ErrorFn,
        GradientFn,
    },
    {    {  new,  (* numOfInputs, numOfHiddenUnits, numOfOutputs *)
            Function[ 
			  Module[ 
			    { 
				  InputSyms,
				  OutputSyms,
				  HiddenUnitSyms,
				  hiddenWeights,
				  hiddenBiases,
				  outputWeights,
				  outputBiases,
				  ParamSymbols,
				  hiddenF,
				  ouputF,
				  errorF,
				  HDTable,
				  DsseDhiddenUnits,
				  DsseDoutputWeights,
				  DsseDoutputBiases,
				  DhiddenUnitsDhiddenWeights,
				  DhiddenUnitsDhiddenBiases,
				  DsseDhiddenWeights,
				  DsseDhiddenBiases,
				  hidWtVals,
				  hidBsVals,
				  outWtVals,
				  outBsVals,
				  i,j
				},
				  
                new[super]; 
             
             (* default constants and arguments *)
                numOfInputs = #1;
                numOfHiddenUnits = #2;
                numOfOutputs = #3;
               
             (* Initialize symbolic input/output matrices *)
                InputSyms  = Array[ Unique[inVar ]&, {1,numOfInputs} ];
                OutputSyms = Array[ Unique[outVar]&, {1,numOfOutputs} ];
                
             (* Initialize symbolic Hidden Unit outputs *)
				HiddenUnitSyms = Array[ Unique[hidUnit]&,{1,numOfHiddenUnits}];
				
             (* Initialize symbolic parameter matrices *)
			    hiddenWeights = Array[ Unique[hidwt]&, {numOfInputs,  numOfHiddenUnits} ];
			    hiddenBiases  = Array[ Unique[hidbs]&, {1, numOfHiddenUnits}            ];
			    outputWeights = Array[ Unique[outwt]&, {numOfHiddenUnits,numOfOutputs} ];
			    outputBiases  = Array[ Unique[outbs]&, {1, numOfOutputs}               ];
				ParamSymbols  = Flatten[{hiddenWeights, hiddenBiases, outputWeights, outputBiases}];
				
            (* Intrinsic Functions *)
                hiddenF = Tanh[ InputSyms . hiddenWeights + hiddenBiases ];
                outputF  = HiddenUnitSyms . outputWeights + outputBiases;
                errorF  = (OutputSyms - outputF) . Transpose[(OutputSyms - outputF)];
              
				HDtable = Dispatch[ Flatten[ 
                    Table[ HiddenUnitSyms[[1,i]] -> hiddenF[[1,i]], {i,numOfHiddenUnits}]
				]];
				
            (* Compile Network Output Function *)
				OutputFn = Compile[ 
				    Release[ Flatten[{ InputSyms, ParamSymbols }]],
					Release[ 
						Return[
						    (outputF /. HDtable)
						]
					]
				];
				
            (* Compile Network Error Function *)
				ErrorFn = Compile[ 
				    Release[ Flatten[{ InputSyms, OutputSyms, ParamSymbols }]],
					Release[ 
						Return[
						    (errorF /. HDtable)
						]
					]
				];
					
            (* Error Gradient Function Expressions *)
                DsseDhiddenUnits           = Table[ D[         errorF, HiddenUnitSyms[[1,i]] ],{i,numOfHiddenUnits}                 ];
                DsseDoutputWeights         = Table[ D[         errorF,  outputWeights[[i,j]] ],{i,numOfHiddenUnits},{j,numOfOutputs}];
                DsseDoutputBiases          ={Table[ D[         errorF,   outputBiases[[1,i]] ],{i, numOfOutputs}                    ]};
				DhiddenUnitsDhiddenWeights = Table[ D[ hiddenF[[1,j]],  hiddenWeights[[i,j]] ],{i,numOfInputs},{j,numOfHiddenUnits} ];
				DhiddenUnitsDhiddenBiases  = Table[ D[ hiddenF[[1,j]],   hiddenBiases[[1,j]] ],{j,numOfHiddenUnits}                 ];
				
                DsseDhiddenWeights         = Table[ DsseDhiddenUnits[[j]] DhiddenUnitsDhiddenWeights[[i,j]],{i,numOfInputs},{j,numOfHiddenUnits}];
                DsseDhiddenBiases          ={Table[ DsseDhiddenUnits[[j]] DhiddenUnitsDhiddenBiases[[j]]   ,{j,numOfHiddenUnits}                ]};
					
            (* Compile Network Gradient Function *)
				GradientFn = List[
					Compile[ Release[ Flatten[{ InputSyms, OutputSyms, ParamSymbols }]],
						Release[ 
							Return[
							    Flatten[Transpose[(DsseDhiddenWeights /. HDtable),{3,4,1,2}],2]
							]
						]
					],
					Compile[ Release[ Flatten[{ InputSyms, OutputSyms, ParamSymbols }]],
						Release[ 
							Return[
							    Flatten[Transpose[( DsseDhiddenBiases /. HDtable),{3,4,1,2}],2]
							]
						]
					],
					Compile[ Release[ Flatten[{ InputSyms, OutputSyms, ParamSymbols }]],
						Release[ 
							Return[
							    Flatten[Transpose[( DsseDoutputWeights /. HDtable),{3,4,1,2}],2]
							]
						]
					],
					Compile[ Release[ Flatten[{ InputSyms, OutputSyms, ParamSymbols }]],
						Release[ 
							Return[
							    Flatten[Transpose[( DsseDoutputBiases /. HDtable),{3,4,1,2}],2]
							]
						]
					]
				];
				
			(* Initialize to random starting parameter values *)
                hidWtVals = Array[ Random[ Statistics`NormalDistribution`NormalDistribution[0,0.1] ]&, {numOfInputs, numOfHiddenUnits}];
                hidBsVals = Array[ Random[ Statistics`NormalDistribution`NormalDistribution[0,0.1] ]&, {1, numOfHiddenUnits}          ];
                outWtVals = Array[ Random[ Statistics`NormalDistribution`NormalDistribution[0,0.1] ]&, {numOfHiddenUnits,numOfOutputs}];
                outBsVals = Array[ Random[ Statistics`NormalDistribution`NormalDistribution[0,0.1] ]&, {1, numOfOutputs}              ];
				Params = {hidWtVals, hidBsVals, outWtVals, outBsVals};
			  ]
			] 
         },
         { Print, (* no arguments *)
           Function[
                Print["numOfInputs:      ", numOfInputs      ];
                Print["numOfHiddenUnits: ", numOfHiddenUnits ];
                Print["numOfOutputs:     ", numOfOutputs     ];
                Print["hiddenWeights: "];   Print[ MatrixForm[ Params[[1]] ]];
                Print["hiddenBiases:  "];   Print[ MatrixForm[ Params[[2]] ]];
                Print["outputWeights: "];   Print[ MatrixForm[ Params[[3]] ]];
                Print["outputBiases:  "];   Print[ MatrixForm[ Params[[4]] ]];
           ]
         },
		 { GetHiddenWeights, (* return hidden weight matrix *)
		   (Params[[1]])&
		 },
		 { GetHiddenBiases, (* return hidden bias matrix *)
		   (Params[[2]])&
		 },
		 { GetOutputWeights, (* return output weight matrix*)
		   (Params[[3]])&
		 },
		 { GetOutputBiases, (* return output bias matrix *)
		   (Params[[4]])&
		 },
		 { GetParameters, (* no arguments *)
		   Params&
		 },
         { N,
           Function[ inputSamples,
                Module[ {i},
                    Return[ 
                        Table[
                            Apply[ 
							    OutputFn, 
								Flatten[{inputSamples[[i]], Params }] 
						    ] [[1]],
                            {i, Length[ inputSamples ]}
                        ]
                    ]
                ]
            ]
         },
         { Error,
           Function[ {inputSamples,outputSamples},
                Module[ {i},
                    Return[ 
                        Sum[
                            Apply[ 
							    ErrorFn, 
								Flatten[{inputSamples[[i]], outputSamples[[i]], Params}]
							][[1]],
                            {i, Length[inputSamples]}
                        ][[1]]
                    ]
                ]
           ]
         },
         { GetOutputFunction,
           OutputFn&
         },
         { GetErrorFunction,
           ErrorFn&
         },
         { GetGradientFunction,
           GradientFn&
         },
         { ErrorGradient,  (* Weight Gradient *)
           Function[ {inputSamples, outputSamples},
               Module[ {i, j},
                  (* Calculate and Return Gradient Sum for all input/output samples *)
                    Return[
                        Sum[ 
                            Table[ 
                                Apply[ 
							        GradientFn[[i]], 
								    Flatten[{inputSamples[[j]], outputSamples[[j]], Params}]
							    ],
                                {i, Length[GradientFn] }
                            ],
                            {j, Length[ outputSamples]}
                        ]
                    ]
                ]
           ]
         },
         { Train,  (* Via Conjugate Gradient Descent *)
           Function[ {inputSamples,outputSamples,trainingEpochs},
               Module[
                   { P0, P1, G0, G1, W0, W1, Beta, Rtn, RtnErrTmp, ErrMin, LearningRate, weightDtable, i, j},
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
                   W1 = Params;

                   For[ i=1, i<=trainingEpochs, i=i+1,
                     (* Prepare for coming iteration *)
                       G0 =. ; G0 = G1;
                       P0 =. ; P0 = P1;
                       W0 =. ; W0 = W1;
                       
                     (* Choose Learning Rate *)
                       ErrMin = Module[ {f, j},
                            f = Function[ x,
                                Sum[
                                    Apply[ 
                                        ErrorFn,
                                        Flatten[ {inputSamples[[j]], outputSamples[[j]], (W0 + x P0)} ]
                                    ][[1]],
                                    {j, Length[outputSamples]}
                                ][[1]]
                            ];
                            FindMinimum[ f[ LearningRate ], {LearningRate, {0.1, 0.2}}]
                       ];
                       
                     (* Update Weights *)
                       W1 = W0 + LearningRate P0 /. ErrMin[[2]];
                       
                     (* Save Weight Values Found *)
					   Params =. ; Params = W1;

                     (* Report Progress and Construct Return List *)       
                       RtnErrTmp = ErrMin[[1]]; 
                       Print[ i, ": ", RtnErrTmp, ", ErrMin = ", ErrMin ];
                       AppendTo[ Rtn,  RtnErrTmp ];
                       
                     (* Find new Gradient *)
                       G1 = ErrorGradient[ self, inputSamples, outputSamples ];
                       
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
	GetOutputFunction,
	GetErrorFunction,
	GetErrorGradientFunction,
	ErrorGradient,
	Train
]

EndPackage[]

(* :History:
   $Log: Perceptron.m,v $
   Revision 1.10  1998/02/27 04:09:59  jak
   A Major Re-write - maybe it'll converge now. -jak

   Revision 1.9  1998/02/26 03:20:41  jak
   I misspelled 'ErrorGradient'. -jak

   Revision 1.8  1998/02/26 02:52:34  jak
   OK ... I had to change the context of the Classes.m for where I
   put it in the Mathematica System to Classes`Classes`. -jak

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
 
