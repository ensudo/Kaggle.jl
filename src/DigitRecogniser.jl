import Flux: onehot, onecold, onehotbatch, normalise, Data, Descent, Dense, Chain, softmax, crossentropy, train!, params, σ
import Calculus
import GLM
import CSV: File, write
import MLLabelUtils: label, labelmap, convertlabel
import DataFrames: first, DataFrame, describe, last, rename!
import StatsPlots: plot, plotlyjs, plot, heatmap, density
import Knet
import StatsBase
import Distributions
import DataFramesMeta
import MultivariateStats
import Statistics: mean
import Missings: skipmissing, replace
import MLDataUtils: splitobs, shuffleobs
import RDatasets: dataset
import LinearAlgebra


plotlyjs()

train = File("../resource/digitrecogniser/train.csv")
test = File("../resource/digitrecogniser/test.csv")
typeof(train)

@assert score > 0.8


