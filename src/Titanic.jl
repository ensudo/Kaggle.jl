import Flux
import Calculus
import GLM
import CSV
import MLLabelUtils
import DataFrames
import Statistics
import StatsPlots
import Knet
import StatsBase
import Distributions
import DataFramesMeta
import Missings

StatsPlots.pyplot()

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

train = CSV.File("./resource/train.csv") |> CSV.DataFrame

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
    Symbol.(MLLabelUtils.labelenc(features) |> MLLabelUtils.label)
end
########################
## Feature Extraction ##
########################

# Data Transformation #

meanage = train[:Age] |> skipmissing |> StatsBase.mean |> round # Mean Age
train[:Age] = Missings.replace(train[:Age], meanage) |> collect # Replace missing with mean Age

meanfare = train[:Fare] |> skipmissing |> StatsBase.mean |> round # Mean Fare
train[:Fare] = Missings.replace(train[:Fare], meanfare) |> collect # Replace missing with mean Fare

modeembarked = train.Embarked |> skipmissing |> StatsBase.mode # Mode Embarked
train.Embarked = Missings.replace(train.Embarked, modeembarked) |> collect # Replace missing Embarked with mode

train[:Fare] = StatsBase.zscore(train[:Fare]) # Scale Fare for outliers 
train[:Age] = StatsBase.zscore(train[:Age]) # Scale Age for outliers

train[:Title] = titles(train[:Name]) # Create new column Title with titles extracted from Name

train[:Sex] = MLLabelUtils.convertlabel([1, -1], train[:Sex]) # 1 = male, -1 = female

titleonehot = MLLabelUtils.convertlabel(
    MLLabelUtils.LabelEnc.OneOfK,train[:Title]
) |> transpose # One-hot encode + reshape matrix to allow being converted to data frame

embarkedonehot = MLLabelUtils.convertlabel(
    MLLabelUtils.LabelEnc.OneOfK, 
    train.Embarked
) |> transpose

embarkedf = DataFrames.names!(
    DataFrames.DataFrame(embarkedonehot), 
    encodinglabels(train.Embarked)
)

titlesdf = DataFrames.names!(
    DataFrames.DataFrame(titleonehot), 
    encodinglabels(train.Title)
)

train = hcat(train, titlesdf, embarkedf) # Concatenate one-hot encoded titles to training DataFrame
DataFrames.deletecols!(train, :Name)   # Remove name as we dont need it
DataFrames.deletecols!(train, :Cabin)  # Remove cabin as it has a high number of missing values
DataFrames.deletecols!(train, :Ticket) # Remove ticket as not enough is known about it

train[[:S, :C, :Q, :Embarked]]

########################
## Exploration        ##
########################


# TODO Go through slides on dimensionality reduction (feature selection)
logreg = GLM.glm(
    GLM.@formula(
        Survived ~ Pclass + Fare + Age + Sex + Mr + Mrs + Miss + Master + Rev + S
    ),
    train, 
    GLM.Binomial(),
    GLM.LogitLink()
)

DataFrames.head(train)
DataFrames.size(train) # shape
DataFrames.describe(train)

## Outliers
StatsPlots.boxplot(hcat(train[:Fare], train[:Age]))
# Mean and standard deviation
faremean = StatsBase.mean(train.Fare)
farestd = StatsBase.std(train.Fare)

# Look how data is distributed
faredist = Distributions.fit(Distributions.Normal, train[:Fare])
agedist = Distributions.fit(Distributions.Normal, train[:Age])

StatsPlots.plot(
    Distributions.Normal(0, 1),
    title = "Age"
) # Start with normal distribution
StatsPlots.density!(train[:Age]) # Overlay Age distribution
Distributions.skewness(train.Age)

StatsPlots.plot(
    Distributions.Normal(0, 1),
    title = "Fare"
) # Start with normal distribution
StatsPlots.density!(train[:Fare]) # Overlay Fare distribution
Distributions.skewness(train.Fare)

train[[:Title, :Sex, :Mr, :Mrs, :Miss, :Master, :Unknown, :Rev]]

StatsPlots.plot(
 StatsPlots.qqplot(
     train[:Fare], 
     train[:Age], 
     qqline = :fit,
     title = "Fare/Age"
    ), # qqplot of two samples, show a fitted regression line
)

StatsPlots.plot(
 StatsPlots.qqplot(
    Distributions.Normal(0,1),
     train[:Fare],
     title = "Fare/Normal"
    ), # qqplot of two samples, show a fitted regression line
)

StatsPlots.plot(
 StatsPlots.qqplot(
    Distributions.Normal(0,1),
     train[:Age],
     title = "Age/Normal"
    ), # qqplot of two samples, show a fitted regression line
)

#=2
Shows difference in people who paid more survived vs people who paid less
tended to not survive. 
=#

StatsPlots.boxplot(train.Survived, train.Fare)
StatsPlots.bar(train.Survived, train.Fare)

train = train[[:Survived, :Pclass, :Sex, :Age, :SibSp, :Parch, :Fare]]
train = DataFrames.dropmissing(train, :Age)

DataFrames.describe(train)
train.Sex = MLLabelUtils.convertlabel([0, 1], train.Sex) # zero = male, female = 1

logreg = GLM.glm(
    GLM.@formula(Survived ~ Pclass),
    train, 
    GLM.Binomial(),
    GLM.LogitLink()
)
#=
Make predictions: Use the intercept (first coeefficienct) 
with the second coefficient (for Pclass)
=#
firstclass, secondclass, thirdclass = 1, 2, 3
Flux.sigmoid(GLM.coef(logreg)[1] + GLM.coef(logreg)[2] * firstclass)
Flux.sigmoid(GLM.coef(logreg)[1] + GLM.coef(logreg)[2] * secondclass)
Flux.sigmoid(GLM.coef(logreg)[1] + GLM.coef(logreg)[2] * thirdclass)
GLM.predict(logreg, DataFrames.DataFrame(Survived = [1], Pclass = [3]))
GLM.predict(logreg)

pred(x) = 1/(1+exp(-(GLM.coef(logreg)[1] + GLM.coef(logreg)[2]*x)))

xGrid = 0:0.1:maximum(train[:Pclass])

StatsPlots.plot(xGrid, pred.(xGrid))

mlogreg = GLM.glm(
    GLM.@formula(Survived ~ Pclass + Sex + Age + Parch + Fare), 
    train, 
    GLM.Binomial(),
    GLM.LogitLink()
)
