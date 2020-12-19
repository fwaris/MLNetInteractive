#load "Packages.fsx"
open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open FSharp.Collections.ParallelSeq
open System
open MathNet.Numerics.Statistics

let ctx = MLContext()

let netflixDataFldr = @"d:\s\ds\netflix"
let dataFiles =  Directory.GetFiles(netflixDataFldr,"combined_data*.txt")

//record to hold parsed required data
type View = 
    {
        Movie : uint32
        Viewer : uint32
        Rating : float32
        Time : DateTime
    }

let parseInt (s:string) = UInt32.Parse(s.Replace(":",""))

//read and parse a single file
let readFile file =
    file
    |> File.ReadLines
    |> Seq.scan (fun st l -> if l.EndsWith(":") then Some l, None else fst st,Some l)  (None,None)
    |> Seq.choose (function 
        | Some m, Some l ->
            let ls = l.Split(',')
            {
                Movie = parseInt m
                Viewer = uint ls.[0]
                Rating = float32 ls.[1]
                Time = match DateTime.TryParse ls.[2] with true,dt -> dt | _ -> failwith "unable to parse"
            }
            |> Some
        | _ -> None
    )

let data = dataFiles |> PSeq.collect readFile      //seq of records from all files (lazy)
let test1 = data |> Seq.take 100 |> Seq.toArray    //view first 100 records to ensure parsing is being done right
let dataView = ctx.Data.LoadFromEnumerable(data)   //convert record seq to IDataView for ML consumption



//view rating distributon
let sampeData = readFile dataFiles.[0] |> Seq.toArray
sampeData.Length
let rng = System.Random(1)
let subSample = sampeData |> Array.filter(fun _ -> rng.NextDouble() < 0.001)
subSample.Length
let rankings = subSample |> Array.map (fun x->float x.Rating) |> Array.countBy (fun x->x)
open FSharp.Plotly
Chart.Column rankings |> Chart.withTitle "Ratings density" |> Chart.Show

//compare rating distributions over time
let minDate = subSample |> Seq.map (fun x->x.Time) |> Seq.min
subSample 
|> Array.groupBy (fun x->x.Rating) 
|> Array.map (fun (r,xs) -> 
    let xs = xs |> Array.sortBy (fun x->x.Time)
    let ratings = xs |> Array.map (fun x-> (x.Time - minDate).TotalDays) 
    let density = xs |> Array.map (fun x -> x.Time, KernelDensity.EstimateGaussian((x.Time-minDate).TotalDays,50.0,ratings))
    r,density)
|> Array.sortBy fst
|> Array.map (fun (r,ds) ->
    Chart.Area ds |> Chart.withTraceName (string r))
|> Chart.Combine
|> Chart.Show





//transform base data to one-hot encode the movie and customer identities (MapValueToKey)
let pairs =  [|"MovieKey","Movie"; "ViewerKey","Viewer"|] |> Array.map (fun (a,b) -> InputOutputColumnPair(a,b))
let txKey=ctx.Transforms.Conversion.MapValueToKey(pairs)
let dataView2 = txKey.Fit(dataView).Transform(dataView)

dataView2.Schema |> Seq.iter (printfn "%A")


//split transfromed data in test and train sets
let tt = ctx.Data.TrainTestSplit(dataView2,testFraction=0.6)


//set options for the recommender model
let opts = Trainers.MatrixFactorizationTrainer.Options()
opts.ApproximationRank <- 150
opts.NumberOfIterations <- 60
opts.MatrixColumnIndexColumnName <- "MovieKey"
opts.MatrixRowIndexColumnName <- "ViewerKey"
opts.LabelColumnName <- "Rating"

//create and train the model
let tr = ctx.Recommendation().Trainers.MatrixFactorization(opts)
let mdl = tr.Fit(tt.TrainSet,tt.TestSet)







