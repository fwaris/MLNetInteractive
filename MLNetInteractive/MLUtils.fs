namespace MLUtils 
open Microsoft.ML
open Microsoft.ML.AutoML
open Microsoft.ML.Data
open System.Collections.Generic

module ColInfo =
    let showCols (colInfo:ColumnInformation) =
        printfn "Column Info"
        printfn "==========="
        printfn $"Categorical: %A{Seq.toArray colInfo.CategoricalColumnNames}"
        printfn $"Numeric:  %A{Seq.toArray colInfo.NumericColumnNames}"
        printfn $"Text:  %A{Seq.toArray colInfo.TextColumnNames}"
        printfn $"Label: %A{colInfo.LabelColumnName}"
        printfn $"Ignored: %A{Seq.toArray colInfo.IgnoredColumnNames}"

    let removeCol (col:string) (colInfo:ColumnInformation) =
        colInfo.CategoricalColumnNames.Remove col |> ignore
        colInfo.TextColumnNames.Remove col |> ignore
        colInfo.NumericColumnNames.Remove col |> ignore
        colInfo.IgnoredColumnNames.Remove col |> ignore
        colInfo.ImagePathColumnNames.Remove col |> ignore

    let ignore (cols:string seq) (colInfo:ColumnInformation) =
        cols |> Seq.iter (fun c -> removeCol c colInfo)
        cols |> Seq.iter (fun c -> colInfo.IgnoredColumnNames.Add c)

    let setAsText (cols:string seq) (colInfo:ColumnInformation) =
        cols |> Seq.iter (fun c -> removeCol c colInfo)
        cols |> Seq.iter (fun c -> colInfo.TextColumnNames.Add c)

    let setAsCategorical (cols:string seq) (colInfo:ColumnInformation) =
        cols |> Seq.iter (fun c -> removeCol c colInfo)
        cols |> Seq.iter (fun c -> colInfo.CategoricalColumnNames.Add c)

    let setAsNumeric (cols:string seq) (colInfo:ColumnInformation) =
        cols |> Seq.iter (fun c -> removeCol c colInfo)
        cols |> Seq.iter (fun c -> colInfo.NumericColumnNames.Add c)

module Schema =
    open System.Collections.Generic

    let printSchema (sch:DataViewSchema) = sch |> Seq.iter (printfn "%A")

    let printIndent indent = for _ in 1 .. indent do  printf " "

    let diffSchemas indent (fromS:DataViewSchema) (toS:DataViewSchema) =
        let fs = fromS |> Seq.map(fun x -> x.Name, string x.Type) |> set        
        let ts = toS |> Seq.map (fun x->x.Name, string x.Type)
        let diff = ts |> Seq.filter (fs.Contains>>not) |> Seq.toList
        diff |> List.iter (fun x -> printIndent indent; printfn "%A" x)

    let rec printTxChain inputSchema indent (root:ITransformer) =
        match root with
        | :? TransformerChain<ITransformer> as tx  -> 
            (inputSchema,tx) 
            ||> Seq.fold(fun inpSch tx ->
                let outSch = tx.GetOutputSchema(inpSch)
                printTxChain inpSch (indent + 1) tx
                outSch)
            |> ignore
        | _ -> 
            printIndent indent
            root.GetType().Name |> printfn "%s"
            let outSch = root.GetOutputSchema (inputSchema)
            diffSchemas (indent + 1) inputSchema outSch

type Viz =
    static member show (dv:IDataView,?title,?rows, ?showHidden) =
        let showHidden = showHidden |> Option.defaultValue false
        let rows = defaultArg rows 1000
        let title = defaultArg title ""
        let rs = dv.Preview(rows).RowView
        let dt = new System.Data.DataTable()
        let _,idx = 
            ((Map.empty,[]),dv.Schema)
            ||> Seq.fold(fun (accMap,accIdx) field -> 
                let n = field.Name
                let accMap,accIdx =
                    match field.IsHidden,showHidden with
                    | true,true 
                    | false,_ ->
                        let acc = accMap |> Map.tryFind n |> Option.map (fun x -> accMap |> Map.add n (x+1)) |> Option.defaultWith(fun _ -> accMap |> Map.add n 0)
                        let count = acc.[n]
                        let fn = if count = 0 then n else $"{n}_{count}"
                        dt.Columns.Add(fn) |> ignore
                        acc,field.Index::accIdx
                    | _ -> accMap,accIdx
                accMap,accIdx)
        let idxSet = HashSet idx
        rs |> Seq.iter (fun r -> 
            let rowVals = r.Values |> Seq.mapi(fun i x -> i,x) |> Seq.choose (fun (i,x) -> if idxSet.Contains i then Some x.Value else None) |> Seq.toArray
            dt.Rows.Add(rowVals) |> ignore)
        let f = new System.Windows.Forms.Form()
        f.Text <- title
        let grid = new System.Windows.Forms.DataGridView()
        grid.Dock <- System.Windows.Forms.DockStyle.Fill
        grid.AutoGenerateColumns <- true
        f.Controls.Add(grid)
        f.Show()
        grid.DataSource <- dt
        grid.Columns 
        |> Seq.cast<System.Windows.Forms.DataGridViewColumn> 
        |> Seq.iter (fun c -> c.SortMode <- System.Windows.Forms.DataGridViewColumnSortMode.Automatic)


//taken from ML.Net fsharp examples
module Pipeline = 
    let downcastPipeline (pipeline : IEstimator<'a>) =
        match pipeline with
        | :? IEstimator<ITransformer> as p -> p
        | _ -> failwith "The pipeline has to be an instance of IEstimator<ITransformer>."

    let append (estimator : IEstimator<'b>) (pipeline : IEstimator<ITransformer>)  = 
           pipeline.Append estimator

    /// combine two transformers
    let inline (<!>) b a = append (downcastPipeline a) (downcastPipeline b)  |> downcastPipeline
