
(* Perceptron.m *)

(* :Title: Single Layer Perceptron Neural Network Class *)

(* :Authors: John A. Kassebaum *)

(* :Summary:
   This class implements a simple perceptron neural network type
   using an object oriented approach in Mathematica.  
 *)
 
(* :Package Version: 1.0 - $Id: Perceptron.m,v 1.2 1998/02/25 07:47:21 jak Exp $ *)

(* :Mathematica Version: 3.0 *)

(* :Copyright: Copyright 1998, John Kassebaum *)

(* :History:
   $Log: Perceptron.m,v $
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

<< ObjectOriented/Classes.m
<< Calculus/VectorCalculus.m
<< Statistics/Master.m

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
         { Grad,  (* Weight Gradient *)
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
                   Print[ "Progress:" ];

                (* Initialize Weight Value Dispatch Table *)
                    weightDtable = Dispatch[{   
                        hiddenWeights -> hidWtVals, 
                        hiddenBiases  -> hidBsVals,
                        outputWeights -> outWtVals, 
                        outputBiases  -> outBsVals
                    }];
                
                (* Initialize Gradients and Directions *)
                   G1 = Grad[ self, inputSamples, outputSamples ];
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
                       RtnErrTmp = ErrMin[[1]]; (* Error[self,inputSamples,outputSamples]; *)
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
];