import Flux: onehot, onecold, onehotbatch, normalise, Data, Descent, Dense, Chain, softmax, crossentropy, train!, params
import Calculus
import GLM
import CSV: File
import MLLabelUtils: label, labelmap, convertlabel
import DataFrames: first, DataFrame, describe
import StatsPlots: plot, plotlyjs, plot, heatmap
import Knet
import StatsBase
import Distributions
import DataFramesMeta
import MultivariateStats
import Statistics: mean
import Missings: skipmissing, replace
import MLDataUtils: splitobs

#[:, start:stride:end]

plotlyjs()

train = File("../resource/train.csv") |> DataFrame

train.Sex = convert(Array{String, 1}, train.Sex)

labels = train[:Survived]
features = train[[:Sex, :Age, :SibSp, :Parch]]

meanage = train.Age |> skipmissing |> mean |> round
features.Age = replace(features.Age, meanage) |> collect
features.Sex = convertlabel([1, -1], features.Sex)

features = Matrix(features) |> transpose

normedfeatures = normalise(features, dims=2)

klasses = unique(labels) |> sort
onehotlabels = onehotbatch(labels, klasses)

# xtrain = splitobs(normedfeatures, at=0.7)[1]
# ytrain = splitobs(onehotlabels, at=0.7)[1]

trainindices = [1:2:891 ; 2:2:891]
xtrain = normedfeatures[:, trainindices]
ytrain = onehotlabels[:, trainindices]


xtest = normedfeatures[:, 3:2:150]
ytest = onehotlabels[:, 3:2:150]


model = Chain(
    Dense(4, 2),# xrows, yrows
    softmax
)

loss(x, y) = crossentropy(model(x), y)
optimiser = Descent(0.5)
dataiterator = Iterators.repeated((xtrain, ytrain), 110)

train!(loss, params(model), dataiterator, optimiser)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

score = accuracy(xtest, ytest)

@assert score > 0.8


function confusionmatrix(X, y)
    ŷ = onehotbatch(onecold(model(X)), 1:3)
    y * ŷ
end

cmat = confusionmatrix(xtest, ytest)

display(cmat)


