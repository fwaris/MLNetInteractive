#load "Packages.fsx"
open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open FSharp.Collections.ParallelSeq
open System

let ctx = MLContext()

let netflixDataFldr = @"D:\s\ds\netflix"
let dataFiles =  Directory.GetFiles(netflixDataFldr,"combined_data*.txt")

dataFiles.[0] |> File.ReadLines |> Seq.take 10 |> Seq.toArray

dataFiles.[0]
|> File.ReadLines
|> Seq.scan (fun st l -> if l.EndsWith(":") then Some l, None else fst st, Some l) (None,None)
|> Seq.choose (function Some m, Some v -> Some (m,v) | _ -> None)
|> Seq.take 10
|> Seq.toArray






//record to hold parsed required data
type View = 
    {
        Movie : uint32
        Viewer : uint32
        Rating : float32
        Time : DateTime
    }

let parseInt (s:string) = UInt32.Parse(s.Replace(":",""))

dataFiles.[0]
|> File.ReadLines
|> Seq.scan (fun st l -> if l.EndsWith(":") then parseInt(l) |> Some, None else (fst st),Some l)  (None,None)
|> Seq.take 10
|> Seq.toArray














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

dataFiles.[0] |> readFile |> Seq.take 10 |> Seq.toArray

