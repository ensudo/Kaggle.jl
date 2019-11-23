import Flux: Data.MNIST, Dense, Conv, Chain, leakyrelu, softmax, softmax, onehotbatch, OneHotMatrix, OneHotVector, relu, maxpool
import Images: load, channelview, ColorTypes, FixedPointNumbers
import FileIO: load
import Base.Iterators: partition
import MLDataUtils: batchview
import Base: Fix1
import StatsPlots: plot, plotlyjs, plot, heatmap, density
import Base: Iterators.PartitionIterator
import Transducers
import MLDataUtils: splitobs

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

@info("loading dataset")
collectedfeatures = loadimages("./resource/digitrecogniser/data")

function minibatch(batchindex::MiniBatchIndex, images::Images)::MiniBatchedImages
    batchindexlength = length(batchindex)
    batch = Array{Float32}(undef, size(images[1])..., 1, batchindexlength)
    for index in 1:batchindexlength
        batch[:, :, :, index] = Float32.(images[batchindex[index]])
    end
    batch
end

function minibatch(batchindex::MiniBatchIndex, labels::Labels)::MiniBatchedLabels
    batch = onehotbatch(labels[batchindex], 0:9)
    batch
end

indexes = partition(1:length(collectedfeatures.images), 128)

batchedimages = [minibatch(index, collectedfeatures.images) for index in indexes]
batchedlabels = [minibatch(index, collectedfeatures.labels) for index in indexes]

xtest, xtrain = splitobs(batchedimages, at=.7, obsdim=ndims(batchedimages))
ytest, ytrain = splitobs(batchedlabels, at=.7, obsdim=ndims(batchedlabels)) # y = what tring to predict (labels)

#trainzipped = NamedTuple{(:label, :image), Tuple{OneHotMatrix{Array{OneHotVector,1}}, Array{Float32, 4}}}.(zip(ytrain, xtrain))
#testzipped = NamedTuple{(:label, :image), Tuple{OneHotMatrix{Array{OneHotVector,1}}, Array{Float32, 4}}}.(zip(ytest, xtest))

model = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), 1=>12, pad=(1, 1), relu),
    x -> maxpool(x, (2, 2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1, 1), relu),
    x -> maxpool(x, (2, 2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1, 1), relu),
    x -> maxpool(x, (2, 2)),

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
    # which is where we get the 288 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10),

    # Finally, softmax to get nice probabilities
    softmax
)

# @assert score > 0.8

