import Flux: onehot, onecold, onehotbatch, normalise, Data, Descent, Dense, Chain, softmax, crossentropy, train!, params
import Calculus
import GLM
import CSV: File
import MLLabelUtils: label, labelmap, convertlabel
import DataFrames: first, DataFrame, describe
import StatsPlots: plot, plotlyjs, plot, heatmap, density
import Knet
import StatsBase
import Distributions
import DataFramesMeta
import MultivariateStats
import Statistics: mean
import Missings: skipmissing, replace
import MLDataUtils: splitobs
import RDatasets: dataset

#[:, start:stride:end]

#=

Labels = vector
Features = transposed vector

start:stride:end = x:y:z

x is where we start, from there we take strides of y, up until but not including z

We split the data x2 train and x2 test sets

onehotlabels dimensions = (number of classes, number of observations)

Iterators.repeated works just like Clojure repeat building a lazy seq

=#

plotlyjs()

data = dataset("datasets", "iris")

train = File("../resource/train.csv") |> DataFrame

train.Sex = convert(Array{String, 1}, train.Sex)

labels = train[:Survived]
features = train[[:Sex, :Age, :SibSp, :Parch]]

meanage = train.Age |> skipmissing |> mean |> round
features.Age = replace(features.Age, meanage) |> collect
features.Sex = convertlabel([1, -1], features.Sex)
features = Matrix(features) |> transpose

normedfeatures = normalise(features, dims=2)
xtrain, xtest = splitobs(
    normedfeatures,
    at=.7,
    obsdim=2
)

klasses = unique(labels) |> sort
onehotlabels = onehotbatch(labels, klasses)
ytrain, ytest = splitobs(
    onehotlabels,
    at=.7,
    obsdim=2
)

uodel = Chain(
    Dense(4, 2),
    softmax
)

loss(x, y) = crossentropy(model(x), y)
optimiser = Descent(.5)
dataiterator = Iterators.repeated((xtrain, ytrain), 110)

train!(loss, params(model), dataiterator, optimiser)

accuracy(x, y) = mean(model(x) |> onecold .== onecold(y))
 # try to predict ytest (labels)  from an input array xtest (features).
score =  accuracy(xtest, ytest)

model(xtest)

#=
The output has two numbers which add up to 1: the probability of not
surviving vs that of surviving. It seems, according to our model, that
this person is unlikely to survive on the titanic.
=#

plot(model(xtest[:, 1]))
model(xtest)[1:2:100]


