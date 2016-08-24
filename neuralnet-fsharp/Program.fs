open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

(*
 *  GENERAL STUFF
 *)

// Types

type Layer = Matrix<double>

type WeightsToNeuron = double list

type WeightsFromInput = double list

type NeuralNet = {
    InputLow        : double;
    InputHigh       : double;
    NumInputs       : int;
    NumHiddenLayers : int;
    NodesPerLayer   : int;
    NumOutputs      : int;
    OutputLow       : double;
    OutputHigh      : double;
    Weights         : Layer list;
}

// Constants

let e = Math.E
let bias = 1.0
let learningRate = 0.3
let rand = new System.Random()

// Functions

let sigmoid (z : double) : Double =
    1.0 / (1.0 + (e ** -z))

let normalize (inputLow   : double)
              (inputHigh  : double)
              (number     : double)
              (outputLow  : double)
              (outputHigh : double) : Double =
    (((number - inputLow) * (outputHigh - outputLow)) / (inputHigh - inputLow)) + outputLow

let weightsToNeuron (layer : Layer) (n : int) : WeightsToNeuron =
    layer.Column(n) |> Vector.toList

let weightsToNextLayer (layer : Layer) : WeightsToNeuron list =
    

let weightsFromInput (layer : Layer) (n : int) : WeightsFromInput =
    layer.Row(n) |> Vector.toList

let weightsFromPreviousLayer (layer : Layer) : WeightsFromInput list =
    

(*
 *  NETWORK CREATION
 *)

let makeLayer numInputs numTargets : Layer =
    CreateMatrix.Dense<double>(numInputs, numTargets, 1.0)

let repeat thing times =
    let rec repeatRec listSoFar times =
        match times with
        | n when n < 0 -> []
        | 0 -> listSoFar
        | _ -> repeatRec (thing :: listSoFar) (times - 1)
    repeatRec [] times

let neuralNet (inputLow        : double)
              (inputHigh       : double)
              (numInputs       : int)
              (numHiddenLayers : int)
              (nodesPerLayer   : int)
              (numOutputs      : int)
              (outputLow       : double)
              (outputHigh      : double) : NeuralNet =
    let inputLayer = makeLayer numInputs nodesPerLayer
    let hiddenWeights = repeat (makeLayer nodesPerLayer nodesPerLayer) (numHiddenLayers - 1)
    let outputLayer = makeLayer nodesPerLayer numOutputs
    {
        NeuralNet.InputLow = inputLow;
        InputHigh = inputHigh;
        NumInputs = numInputs;
        NumHiddenLayers = numHiddenLayers;
        NodesPerLayer = nodesPerLayer;
        NumOutputs = numOutputs;
        OutputLow = outputLow;
        OutputHigh = outputHigh;
        Weights = (inputLayer :: hiddenWeights) @ [ outputLayer ]
    }

let randWeight (r : int) (c : int) (d : double) (t : 'a) =
    rand.NextDouble()

let randomizeNet (network : NeuralNet) : NeuralNet =
    let randomizedWeights =
        network.Weights
        |> List.map (fun l -> l.MapIndexed (fun (r:int) (c:int) (d:double) -> rand.NextDouble()))
    { network with Weights = randomizedWeights }

(*
 *  UTILIZATION FUNCTIONS
 *)

let activate (inputs : double list) (weightsToNeuron : WeightsToNeuron) : double =
    List.map2 (fun f s -> f * s) inputs weightsToNeuron
    |> List.sum
    |> sigmoid

let activateLayer (inputs : double list) (layer : Layer) : double list =
    

[<EntryPoint>]
let main argv =
    printfn "%A" argv
    0
