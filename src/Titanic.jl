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


#[:, start:stride:end]

#=
d
Labels = vector
Features = transposed vector

start:stride:end = x:y:z

x is where we start, from there we take strides of y, up until but not including z

We split the data x2 data and x2 test sets

onehotlabels dimensions = (number of classes, number of observations)

Iterators.repeated works just like Clojure repeat building a lazy seq

Accuracy was increased from .76 to .83 by randomising the dataset

TODO Try get above .9 by adding more layers

=#

plotlyjs()

data = File("resource/titanic/data.csv") |> DataFrame

# data = shuffleobs(data)
data.Sex = convert(Array{String, 1}, data.Sex)

labels = data[:Survived]
features = data[[:Sex, :SibSp, :Age, :Parch, :Fare, :Pclass]]

meanage = data.Age |> skipmissing |> mean |> round
features.Age = replace(features.Age, meanage) |> collect

meanfare = data.Fare |>  skipmissing |> mean |> round
features.Fare = replace(features.Fare, meanfare) |> collect

features.Sex = convertlabel([1, -1], features.Sex)

features = Matrix(features) |> transpose

features = convert(LinearAlgebra.Transpose{Float64, Array{Float64, 2}}, features)

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

#=
Declare model taking 5 features as inputs and outputting
2 probabilities.. one for probability of surviving
and one for probability of dying.
=#

featuresrows,_ = size(features)
lenclasses = length(klasses)

model = Chain(
    Dense(featuresrows, lenclasses),
    softmax
)

# model = Chain(
#     Dense(featuresrows, featuresrows, σ),
#     Dense(featuresrows, featuresrows, σ),
#     Dense(featuresrows, lenclasses),
#     softmax
# )

loss(x, y) = crossentropy(model(x), y)

optimiser = Descent(0.5)

dataiterator = Iterators.repeated((xtrain, ytrain), 110) #110 epochs

train!(loss, params(model), dataiterator, optimiser)

accuracy(x, y) = mean(model(x) |> onecold .== onecold(y))

score =  accuracy(xtest, ytest) # Try to predict ytest (labels) from an input array xtest (features).

@assert score > 0.8

function confusionmatrix(X, y)
    ŷ = onehotbatch(onecold(model(X)), 1:lenclasses)
    y * ŷ'
end

model(xtest)

#=
The output has two numbers which add up to 1: the probability of not
surviving vs that of surviving. It seems, according to our model, that
this person is unlikely to survive on the titanic.
=#

model(xtest[:, 1])
model(xtest)[1:2:100]

display(confusionmatrix(xtest, ytest))


