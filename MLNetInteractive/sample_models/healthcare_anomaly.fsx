#load "../Packages.fsx"

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.AutoML
open MLUtils
open MLUtils.Pipeline
open FSharp.Data
open FSharp.Plotly

let ctx = MLContext()

//dataset downloaded from here:
//https://www.kaggle.com/tamilsel/healthcare-providers-data
[<Literal>]
let inputFile = @"D:\s\mlnetconf\Healthcare Providers.csv"

type Thp = FSharp.Data.CsvProvider< inputFile>
let thp = Thp.GetSample()
let thp_rs = thp.Rows |> Seq.toArray

[
    thp_rs |> Array.map (fun x->x.``Average Medicare Allowed Amount``) |> Chart.Violin |> Chart.withTraceName "Avg. Medicare Allowed"
    thp_rs |> Array.map (fun x->x.``Average Submitted Charge Amount``) |> Chart.Violin |> Chart.withTraceName "Avg. Submitted Charge"
    thp_rs |> Array.map (fun x->x.``Average Medicare Payment Amount``) |> Chart.Violin |> Chart.withTraceName "Avg. Medicare Payment"
    thp_rs |> Array.map (fun x->x.``Average Medicare Standardized Amount``) |> Chart.Violin |> Chart.withTraceName "Avg. Medicare Standardized"
]
|> Chart.Combine
|> Chart.withLayout(Layout.init(Margin=Margin.init(Left=300)))
|> Chart.withSize(1000.,700.)
|> Chart.Show   

[
    thp_rs |> Array.map (fun x->x.``Number of Distinct Medicare Beneficiary/Per Day Services``) |> Chart.BoxPlot |> Chart.withTraceName "Medicare Beneficiary/Per Day Services"
    thp_rs |> Array.map (fun x->x.``Number of Medicare Beneficiaries``) |> Chart.Violin |> Chart.withTraceName "Medicare Beneficiaries"
    thp_rs |> Array.map (fun x->x.``Number of Services``) |> Chart.Violin |> Chart.withTraceName "Services"
]
|> Chart.Combine
|> Chart.withMarginSize(Left = 300)
|> Chart.withSize(1000.,700.)
|> Chart.Show   

thp_rs 
|> Array.map (fun x->x.``HCPCS Drug Indicator``)
|> Chart.Histogram 
|> Chart.withTitle "Histogram" 
|> Chart.withMarginSize(Bottom = 300)
|> Chart.withSize(1000.,700.)
|> Chart.Show


let colInfoR = ctx.Auto().InferColumns(inputFile, labelColumnName="index", groupColumns=false)
let ci = colInfoR.ColumnInformation
ColInfo.showCols ci
ci |> ColInfo.ignore [
    "index"
    "First Name of the Provider"
    "Middle Initial of the Provider"
    "Credentials of the Provider"
    "Street Address 2 of the Provider"
    "Country Code of the Provider"
    "HCPCS Code"
    "HCPCS Description"
    "National Provider Identifier"
    "Zip Code of the Provider"
    ]
ci |> ColInfo.setAsCategorical ["Medicare Participation Indicator"; "HCPCS Drug Indicator"]
ci |> ColInfo.setAsNumeric 
    [
        "Average Medicare Allowed Amount"
        "Average Medicare Payment Amount"
        "Average Medicare Standardized Amount"
        "Number of Services"
        "Number of Distinct Medicare Beneficiary/Per Day Services";
        "Average Submitted Charge Amount" 
        "Number of Medicare Beneficiaries"
    ]

let dv = ctx.Data.LoadFromTextFile(inputFile, colInfoR.TextLoaderOptions)
Viz.show dv
dv.Schema |> Schema.printSchema

ci.CategoricalColumnNames |> Seq.toArray

let txToSingle = 
    let colPairs = ci.NumericColumnNames |> Seq.map(fun x -> InputOutputColumnPair(x,x)) |> Seq.toArray
    ctx.Transforms.Conversion.ConvertType(colPairs, DataKind.Single)
let oneHTxs = ci.CategoricalColumnNames |> Seq.map(fun c-> ctx.Transforms.Categorical.OneHotEncoding(c) |> Pipeline.downcastPipeline)
let txOneH = oneHTxs |> Seq.reduce (<!>)
let ftrCols = Seq.append ci.CategoricalColumnNames ci.NumericColumnNames |> Seq.toArray
let txFtrs = ctx.Transforms.Concatenate("Features",ftrCols)
let txD = txToSingle <!> txOneH <!> txFtrs
let dvTrain = txD.Fit(dv).Transform(dv)
// Viz.show dvTrain

let topts = Trainers.RandomizedPcaTrainer.Options()
topts.FeatureColumnName <- "Features"
topts.Rank <- 3
let trainer = ctx.AnomalyDetection.Trainers.RandomizedPca(topts)
let mdl = trainer.Fit(dvTrain)
let dvScore = mdl.Transform(dvTrain)
// Viz.show dvScore

// Schema.printSchema dvScore.Schema

[<CLIMutable>]
type TScore = 
    {
        index:float32
        Score:float32
        ``Number of Medicare Beneficiaries``                         : single
        ``Average Medicare Allowed Amount``                          : single
        ``Average Medicare Payment Amount``                          : single
        ``Average Medicare Standardized Amount``                     : single
        ``Number of Services``                                       : single
        ``Number of Distinct Medicare Beneficiary/Per Day Services`` : single
        ``Average Submitted Charge Amount``                          : single
        }

let scores = ctx.Data.CreateEnumerable<TScore>(dvScore,false)

let scoreChart t (xs:(float32*float32) seq) =
    let maxX = xs |> Seq.map fst |> Seq.max
    [
        xs |> Chart.Point |> Chart.withTraceName "Data"
        [0.0f,0.7f; maxX,0.7f] |> Chart.Line |> Chart.withTraceName "Threshold"
    ]
    |> Chart.Combine
    |> Chart.withY_AxisStyle("Anomaly score")
    |> Chart.withTitle t
    |> Chart.Show
    
let rs = scores |> Seq.map (fun x->x.Score) |> Seq.toArray

scores |> Seq.map(fun x -> x.``Number of Distinct Medicare Beneficiary/Per Day Services``,x.Score) |> scoreChart "Number of Distinct Medicare Beneficiary/Per Day Services"
scores |> Seq.map(fun x -> x.``Number of Services``,x.Score) |> scoreChart "Average Submitted Charge Amount" 
scores |> Seq.map(fun x -> x.``Average Medicare Allowed Amount``,x.Score) |> scoreChart "Average Medicare Allowed Amount" 
scores |> Seq.map(fun x -> x.``Average Medicare Standardized Amount``,x.Score) |> scoreChart "Average Medicare Standardized Amount" 
scores |> Seq.map(fun x -> x.``Average Medicare Payment Amount``,x.Score) |> scoreChart "Average Medicare Payment Amount" 
scores |> Seq.map(fun x -> x.``Average Submitted Charge Amount``,x.Score) |>  scoreChart "Average Submitted Charge Amount"


let highAnomaly = scores |> Seq.filter (fun x -> x.Score > 0.8f) |> Seq.toArray

let dvF = ctx.Data.FilterRowsByColumn(dvScore,"Score",lowerBound=0.8)
Viz.show dvF















