open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

let e = Math.E
let bias = 1.0
let learningRate = 0.3

let sigmoid (z : double) =
    1.0 / (1.0 + (e ** -z))

let normalize (inputLow   : double)
              (inputHigh  : double)
              (number     : double)
              (outputLow  : double)
              (outputHigh : double) =
    (((number - inputLow) * (outputHigh - outputLow)) / (inputHigh - inputLow)) + outputLow

type Layer = Matrix<double>

let weightsToNeuron (layer : Layer) (n : int) =
    layer.Column(n)

let weightsFromInput (layer : Layer) (n : int) =
    layer.Row(n)

type NeuralNet = {
    InputLow        : double;
    InputHigh       : double;
    NumInputs       : int;
    NumHiddenLayers : int;
    NumLayerNodes   : int;
    NumOutputs      : int;
    OutputLow       : double;
    OutputHigh      : double;
    Weights         : List<Layer>;
}

let makeLayer numInputs numTargets : Layer =
    CreateMatrix.Dense<double>(numInputs, numTargets, 1.0)

[<EntryPoint>]
let main argv =
    printfn "%A" argv
    0
