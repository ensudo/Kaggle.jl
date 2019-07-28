import Flux: Data.MNIST, Dense, leakyrelu, onehotbatch
import Images: load, channelview, ColorTypes, FixedPointNumbers
import FileIO: load
import Base.Iterators: partition
import MLDataUtils: batchview
import Base: Fix1
import StatsPlots: plot, plotlyjs, plot, heatmap, density
import Base: Iterators.PartitionIterator
# TODO put types in seperate file
# TODO Minibatch

plotlyjs()

parseint = Fix1(parse, Int)

GreyImage = Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}}, 2}
MiniBatchIndex = UnitRange{Int64}
MiniBatchIndexes = PartitionIterator{MiniBatchIndex}
Label = Int64
BatchSize = Int64
Labels = Array{Label, 1}
Images = Array{GreyImage, 1}

struct ImageFeatures
    labels::Labels
    images::Images
end

function loadimages(path::String)::ImageFeatures
    images::Array{GreyImage, 1} = []
    labels::Array{Label, 1}  = []
    nothidden = file -> !startswith(file, ".")
    files = filter(nothidden, readdir(path))
    for file in files
        imagepath = string(path, "/", file)
        image = load(imagepath)
        label = split(file, ".") |> first |> last |> parseint
        push!(labels, label)
        push!(images, image)
    end
    ImageFeatures(labels, images)
end

features = loadimages("../resource/digitrecogniser/data")

function minibatch(images::Images, batchindex::MiniBatchIndex)
    batch = Array{Float32}(
        undef,
        size(images[1])...,
        1,
        length(batchindex)
    )
    for index in 1:length(batchindex)
        batch[:, :, :, index] = Float32.(images[batchindex[index]])
    end
    return batch
end

function minibatch(labels::Labels, batchindex::MiniBatchIndex)
    batch = onehotbatch(labels[batchindex], 0:9)
    batch
end

indexes = partition(1:length(features.images), 128 )

trainimages = [minibatch(features.images, index) for index in indexes]
trainlabels = [minibatch(features.labels, index) for index in indexes]

# TODO investigate multiple dispatch

a2 = ((1, 2), (3, 4), (5, 6))
a3 = zip([1, 2, 3], [4, 5, 6])
a2 |> collect
a3 |> collect

batchview(data)

@assert score > 0.8

################################################################################

train_labels = MNIST.labels()
train_imgs = MNIST.images()
train_imgs[1]
# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end
batch_size = 128
mb_idxs = partition(1:length(train_imgs), batch_size)
train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

################################################################################
train_imgs
features.images
