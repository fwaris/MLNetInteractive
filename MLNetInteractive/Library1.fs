module MLNetInteractive
open Microsoft.ML
open System
open Microsoft.ML.Data

let check() =
    let mutable vb = VBuffer(1,[|2.0f|])
    let mutable m = vb.GetValues()
    m.ToArray()

let m = check()

System.Console.ReadLine() |> ignore

