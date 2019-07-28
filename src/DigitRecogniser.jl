import Flux: Data.MNIST, Dense, leakyrelu, onehotbatch, OneHotMatrix, OneHotVector
import Images: load, channelview, ColorTypes, FixedPointNumbers
import FileIO: load
import Base.Iterators: partition
import MLDataUtils: batchview
import Base: Fix1
import StatsPlots: plot, plotlyjs, plot, heatmap, density
import Base: Iterators.PartitionIterator

plotlyjs()

parseint = Fix1(parse, Int)

GreyImage = Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}}, 2}
Images = Array{GreyImage, 1}
MiniBatchIndex = Array{Int64, 1}
Label = Int64
Labels = Array{Label, 1}
MiniBatchedImages = Array{Float32, 4}
MiniBatchedLabels = OneHotMatrix{Array{OneHotVector,1}}

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

function minibatch(batchindex::MiniBatchIndex, images::Images)::MiniBatchedImages
    batch = Array{Float32}(
        undef,
        size(images[1])...,
        1,
        length(batchindex)
    )
    for index in 1:length(batchindex)
        batch[:, :, :, index] = Float32.(images[batchindex[index]])
    end
    batch
end

function minibatch(batchindex::MiniBatchIndex, labels::Labels)::MiniBatchedLabels
    batch = onehotbatch(labels[batchindex], 0:9)
    batch
end

indexes = partition(1:length(features.images), 128)
trainimages = [minibatch(index, features.images) for index in indexes]
trainlabels = [minibatch(index, features.labels) for index in indexes]

@assert score > 0.8
