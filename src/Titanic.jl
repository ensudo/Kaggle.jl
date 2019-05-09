import Flux
import Calculus
import GLM
import CSV
import MLLabelUtils
import DataFrames
import StatsPlots
import Knet
import StatsBase
import Distributions
import DataFramesMeta
import MultivariateStats
import Statistics
import Missings

const flux              = Flux
const tracker           = Flux.Tracker
const calculus          = Calculus
const glm               = GLM
const csv               = CSV
const statsplots        = StatsPlots
const multivariatestats = MultivariateStats
const mllabelutils      = MLLabelUtils
const dataframes        = DataFrames
const statistics        = Statistics
const distributions     = GLM.Distributions
const knet              = Knet
const statsbase         = StatsBase
const missings          = Missings

statsplots.pyplot(width = 100)

∑ = +

#=

Classification Models
---------------------

Logistic Regression
Multiple Logistic Regression
Probit regression
OLS regression
Two-group discriminant function analysis
Hotelling’s T2
Linear Discriminant Analysis
K-nearest neighbours
Generalized Additive Models
Trees
Random Forest
Boosting
Support Vector Machines

=#

# TODO read!! http://ww2.coastal.edu/kingw/statistics/R-tutorials/logistic.html
# TODO analysis on what predictors are most relevent to predicting the response variable
# TODO Research how to count missing values

train = CSV.File("./resource/train.csv") |> dataframes.DataFrame

function titles(xs::Array{Union{Missing, String}, 1})
    acc::Array{String, 1} = []
    for x in xs
        if occursin("Mr.", x)
            push!(acc, "Mr")
        elseif occursin("Mrs.", x)
            push!(acc, "Mrs")
        elseif occursin("Miss.", x)
            push!(acc, "Miss")
        elseif occursin("Master.", x)
            push!(acc, "Master")
        elseif occursin("Rev.", x)
            push!(acc, "Rev")
        else push!(acc, "Unknown")
        end
    end
    acc
end

function encodinglabels(features::Array{String, 1})
    Symbol.(mllabelutils.labelenc(features) |> mllabelutils.label)
end
########################
## Feature Extraction ##
########################

# Data Transformation #

meanage = train[:Age] |> skipmissing |> statistics.mean |> round # Mean Age
train[:Age] = Missings.replace(train[:Age], meanage) |> collect # Replace missing with mean Age

meanfare = train[:Fare] |> skipmissing |> statistics.mean |> round # Mean Fare
train[:Fare] = Missings.replace(train[:Fare], meanfare) |> collect # Replace missing with mean Fare

modeembarked = train.Embarked |> skipmissing |> statsbase.mode # Mode Embarked
train.Embarked = Missings.replace(train.Embarked, modeembarked) |> collect # Replace missing Embarked with mode

train[:Fare] = glm.zscore(train[:Fare]) # Scale Fare for outliers
train[:Age] = glm.zscore(train[:Age]) # Scale Age for outliers

train[:Title] = titles(train[:Name]) # Create new column Title with titles extracted from Name

train[:Sex] = mllabelutils.convertlabel([1, -1], train[:Sex]) # 1 = male, -1 = female

embarkedonehot = mllabelutils.convertlabel(
    mllabelutils.LabelEnc.OneOfK,
    train.Embarked
) |> transpose

embarkedf = dataframes.names!(
    dataframes.DataFrame(embarkedonehot),
    encodinglabels(train.Embarked)
)

titleonehot = mllabelutils.convertlabel(
    mllabelutils.LabelEnc.OneOfK,train[:Title]
) |> transpose # One-hot encode + reshape matrix to allow being converted to data frame

titlesdf = dataframes.names!(
    dataframes.DataFrame(titleonehot),
    encodinglabels(train.Title)
)

titlesdf = dataframes.names!(
    dataframes.DataFrame(titleonehot),
    encodinglabels(train.Title)
)

train = hcat(train, titlesdf, embarkedf) # Concatenate one-hot encoded titles to training DataFrame
dataframes.deletecols!(train, :Name)   # Remove name as we dont need it
dataframes.deletecols!(train, :Cabin)  # Remove cabin as it has a high number of missing values
dataframes.deletecols!(train, :Ticket) # Remove ticket as not enough is known about it

########################
## Exploration        ##
########################

# TODO Go through slides on dimensionality reduction (feature selection)


# Remove any high;y correlated variables

#vDrvp highly correlated features from the model

corrdf = train[
    [
        :Survived,
        :Pclass,
        :Sex,
        :Age,
        :SibSp,
        :Parch,
        :Fare,
        :Mr,
        :Mrs,
        :Miss,
        :Master,
        :Unknown,
        :Rev,
        :S,
        :C,
        :Q
    ]
]

#=
Can you correlate categorical variables?
Without order, it's not possible to correlate two variables. But never fear, there
are ways to find out if categorical variables are related in some way; you need to
simply move from correlation to association. These would be tests such
as Chi square and ANOVAs.
=#

statsplots.crosscor(train.S, train.C)

cormat = convert(
    Array{Float64, 2},
    corrdf
) |> distributions.cor

statsplots.corrplot(cormat, size = (2000, 1500), label = names(corrdf))

statsplots.@df corrdf statsplots.boxplot(
    [:Rev :S :C :Q],
    size = (2000, 1500),
)

# TODO read slides on dimensiality redection

#=
Rev+S+C+Q <- TODO find whats up with these.. PCA will fix this but first look
into finding out which variables are the culprits of failing colinearality when model is fit
=#
logreg = glm.glm(
    glm.@formula(
        Survived ~ Pclass+Sex+Age+SibSp+Parch+Fare+Mr+Mrs+Miss+Master+Unknown
    ),
    corrdf,
    glm.Binomial(),
    glm.LogitLink()
)


dataframes.head(train)
size(train) # shape
dataframes.describe(train)

## Outliers
statsplots.boxplot(hcat(train[:Fare], train[:Age]))
# Mean and standard deviation
faremean = statistics.mean(train.Fare)
farestd = statistics.std(train.Fare)

# Look how data is distributed
faredist = glm.fit(distributions.Normal, train[:Fare])
agedist = glm.fit(distributions.Normal, train[:Age])

statsplots.plot(
    distributions.Normal(0, 1),
    title = "Age"
) # Start with normal distribution
statsplots.density!(train[:Age]) # Overlay Age distribution
distributions.skewness(train.Age)

statsplots.plot(
    distributions.Normal(0, 1),
    title = "Fare"
) # Start with normal distribution
statsplots.density!(train[:Fare]) # Overlay Fare distribution
distributions.skewness(train.Fare)

train[[:Title, :Sex, :Mr, :Mrs, :Miss, :Master, :Unknown, :Rev]]

statsplots.plot(
    statsplots.qqplot(
     train[:Fare],
     train[:Age],
     qqline = :fit,
     title = "Fare/Age"
    ), # qqplot of two samples, show a fitted regression line
)

statsplots.plot(
    statsplots.qqplot(
    distributions.Normal(0,1),
     train[:Fare],
     title = "Fare/Normal"
    ), # qqplot of two samples, show a fitted regression line
)

statsplots.plot(
    statsplots.qqplot(
    distributions.Normal(0,1),
     train[:Age],
     title = "Age/Normal"
    ), # qqplot of two samples, show a fitted regression line
)
