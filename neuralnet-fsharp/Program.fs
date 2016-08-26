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

type TrainingInstance = {
    In  : double list;
    Out : double list;
}

type TrainingSet = TrainingInstance list

// Constants and Objects

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

let reverseList list = List.fold (fun acc elem -> elem :: acc) [] list

let averageDoubles (listOfDouble : double list) = List.sum listOfDouble / (double listOfDouble.Length)

let dotProductList (list1 : double list) (list2 : double list) : double =
    List.map2 (*) list1 list2 |> List.sum

let dotProductVec (vec1 : Vector<double>) (vec2 : Vector<double>) : double =
    Vector.map2 (*) vec1 vec2 |> Vector.sum

let weightsToNeuron (layer : Layer) (n : int) : WeightsToNeuron =
    layer.Column(n) |> Vector.toList

let weightsToNextLayer (layer : Layer) : WeightsToNeuron list =
    layer.EnumerateColumns() 
    |> Seq.toList
    |> List.map Vector.toList
    //List.map (fun c -> layer.Column(c) |> Vector.toList) [0..layer.ColumnCount]

let rowsFromColumns (columns : WeightsToNeuron list) : WeightsFromInput list =
    let numRows = columns.Head.Length
    List.map (fun rowNum -> List.map (fun (col : WeightsToNeuron) -> col.Item(rowNum)) columns) [0..numRows - 1]

let weightsFromInput (layer : Layer) (n : int) : WeightsFromInput =
    layer.Row(n) |> Vector.toList

let weightsFromPreviousLayer (layer : Layer) : WeightsFromInput list =
    layer.EnumerateRows()
    |> Seq.toList
    |> List.map Vector.toList
    //List.map (fun r -> layer.Row(r) |> Vector.toList) [0..layer.RowCount]

(*
 *  NETWORK CREATION
 *)

let makeLayer numInputs numTargets : Layer =
    CreateMatrix.Dense<double>(numInputs, numTargets, 1.0)

let repeat (thing : 'T) times : 'T list =
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
    let hiddenLayers = repeat (makeLayer nodesPerLayer nodesPerLayer) (numHiddenLayers - 1)
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
        Weights = (inputLayer :: hiddenLayers) @ [ outputLayer ]
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
    dotProductList inputs weightsToNeuron
    |> sigmoid

let activateLayer (inputs : double list) (layer : Layer) : double list =
    weightsToNextLayer layer
    |> List.map (fun n -> activate inputs n)

let evaluate (inputs : double list) (network : NeuralNet) : double list =
    let rec feedForward (prevVals : double list) (remainingLayers : Layer list) : double list =
        match remainingLayers with
        | [] -> prevVals
                |> List.map (fun v -> normalize -1.0
                                                1.0 
                                                v 
                                                network.OutputLow 
                                                network.OutputHigh)
        | first :: rest -> feedForward (activateLayer prevVals first)
                                       rest
    feedForward (List.map (fun i -> normalize network.InputLow
                                              network.InputHigh
                                              i
                                              -1.0
                                              1.0) inputs)
                network.Weights

(*
 *  TRAINING FUNCTIONS
 *)

let meanSquaredError (actual : double) (expected : double) : double =
    learningRate * ((actual - expected) ** 2.0)

let neuronError (prevErrors : double list) (neuronWeights : WeightsFromInput) : double =
    dotProductList prevErrors neuronWeights

let layerErrors (prevErrors : double list) (layer : Layer) : double list =
    List.map (fun n -> neuronError prevErrors n) (weightsFromPreviousLayer layer)

let adjustWeightsToNeuron (neuronError : double) (weightsToNeuron : WeightsToNeuron) : WeightsToNeuron =
    List.map (fun w -> w - neuronError) weightsToNeuron

let adjustLayerWeights (layerErrors : double list) (layer : Layer) : Layer =
    List.map2 adjustWeightsToNeuron layerErrors (weightsToNextLayer layer)
    |> rowsFromColumns
    |> matrix

let adjustNetWeights (errors : double list list) (network : NeuralNet) : NeuralNet =
    let newWeights = List.map2 adjustLayerWeights errors network.Weights
    { network with Weights = newWeights }

let trainInstance (instance : TrainingInstance) (network : NeuralNet) : NeuralNet =
    let rec backPropagate (errors : double list list) (layersLeft : Layer list) : NeuralNet =
        match layersLeft with
        | [] -> adjustNetWeights errors network
        | first :: rest -> backPropagate ((layerErrors errors.Head first) :: errors) rest
    backPropagate [(List.map2 meanSquaredError (evaluate instance.In network) instance.Out)]
                  (reverseList network.Weights)

let rec trainSet (set : TrainingSet) (network : NeuralNet) : NeuralNet =
    match set with
    | [] -> network
    | first :: rest -> trainSet (rest) (trainInstance first network)

let averageEpochError (set : TrainingSet) (network : NeuralNet) : double =
    set
    |> List.map (fun (i : TrainingInstance) -> List.map2 meanSquaredError (evaluate i.In network) i.Out
                                               |> averageDoubles)
    |> averageDoubles

let fullyTrainNetwork (set: TrainingSet) (errorFloor : double) (network : NeuralNet) : NeuralNet =
    let rec epoch (net : NeuralNet) (epochs : int) : NeuralNet =
        if epochs < 1000 then
            printfn "%f" (averageEpochError set net)
            epoch (trainSet set net) (epochs + 1)
        else
            net
    epoch network 1
    
[<EntryPoint>]
let main argv =
    let andData : TrainingSet = [{ In = [0.0; 0.0]; Out = [0.0]; };
                                 { In = [1.0; 0.0]; Out = [0.0]; };
                                 { In = [0.0; 1.0]; Out = [0.0]; };
                                 { In = [1.0; 1.0]; Out = [1.0]; };]
    let andGate : NeuralNet =
        neuralNet 0.0 1.0 2 2 2 1 0.0 1.0
        |> randomizeNet
        |> fullyTrainNetwork andData 0.05
    0
