﻿#r "nuget: Microsoft.ML.AutoML, Version=0.17.2" 
#r "nuget: FSharp.Collections.ParallelSeq"
#r "nuget: Microsoft.ML.Recommender"
#r "nuget: FSharp.Plotly"
#r "nuget: MathNet.Numerics.FSharp"

// workaround till native library transitive references are fixed for F# interactive
let userProfile = System.Environment.GetEnvironmentVariable("UserProfile")
let packageRoot = $@"{userProfile}\.nuget\packages"
let nativeLib =  $@"{packageRoot}\microsoft.ml.cpumath\1.5.2\runtimes\win-x64\nativeassets\netstandard2.0"//CpuMathNative.dll"
let nativeLib2 = $@"{packageRoot}\microsoft.ml.recommender\0.17.2\runtimes\win-x64\native"//MatrixFactorizationNative.dll"
let path = System.Environment.GetEnvironmentVariable("path")
let path' =  path + ";" + nativeLib + ";" + nativeLib2
System.Environment.SetEnvironmentVariable("path",path')





