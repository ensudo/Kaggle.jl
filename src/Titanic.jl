import Flux: onehot, onehotbatch, normalise
import Calculus
import GLM
import CSV: File
import MLLabelUtils: label, labelmap, convertlabel
import DataFrames: first, DataFrame, describe
import StatsPlots: plot
import Knet
import StatsBase
import Distributions
import DataFramesMeta
import MultivariateStats
import Statistics: mean
import Missings: skipmissing, replace


statsplots.plotlyjs()

train = File("../resource/train.csv") |> DataFrame

train.Sex = convert(Array{String, 1}, train.Sex)

meanage = train.Age |> skipmissing |> mean |> round

train.Age = replace(train.Age, meanage) |> collect

train.Sex = convertlabel([1, -1], train.Sex)



first(train, 10)

describe(train)


