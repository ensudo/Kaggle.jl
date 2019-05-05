
survivedcount = count(x -> x == 1,  train.Survived)
diedcount = count(x -> x == 0,  train.Survived)

statsplots.histogram(group = mllabelutils.convertlabel(["Died", "Survived"], train.Survived), train.Fare)


mllabelutils.convertlabel(["Died", "Survived"], train.Survived)

ctg = repeat(["Category 1", "Category 2"], inner = 5)

statsplots.@df train[[:Survived, :Fare]] statsplots.boxplot(:Survived, :Fare)

mllabelutils.convertlabel(["Died", "Survived"], train.Survived)

#=
Why no fstatistic?

GeneralizedLinearModels do support an analysis of deviance, which is similar in 
spirit to analysis of variance but, I believe, is equivalent to 
a likelihood ratio test
=#

glm.deviance(logreg)

logreg
glm.confint(logreg)
glm.confint(mlogreg)
train[:Prediction] = glm.predict(logreg)
diedsurvived = mllabelutils.convertlabel(["Died", "Survived"], train.Survived)
flux.sigmoid(-3.5041 + 0.4049) # Page 135 of Introduction to Statistical Learning

distributions.Normal(statsbase.mean(x), statsbase.std(x))


gendersubmission = csv.File("./resource/gender_submission.csv") |> csv.DataFrame
train = csv.File("./resource/train.csv") |> csv.DataFrame
test = csv.File("./resource/test.csv") |> csv.DataFrame
menache = csv.File("~/Data/menarche.csv") |> dataframes.DataFrame

menache[:Total] = convert(Array{Int, 1}, menache[:Total])
xGrid = 0:0.1:maximum(menache[:Age])

logreg = glm.glm(
    glm.@formula(Total ~ Age),
    menache,
    glm.Binomial(),
    glm.LogitLink()
)

statsplots.plot(xGrid, glm.predict(logreg))


########################
## Feature Extraction ##
########################

dataframes.head(train)
dataframes.size(train) # shape
dataframes.describe(train)

train[:Fare] = convert(Array{Float64, 1}, train[:Fare])

## Outliers

faremean = statsbase.mean(train.Fare)
farestd = statsbase.std(train.Fare)

# Estimate mean and standard deviation
faredist = distributions.fit(distributions.Normal, train.Fare)
# View distribution
statsplots.plot(faredist)
# View outliers
statsplots.boxplot(train.Fare)
statsbase.iqr(train.Fare)

statsplots.boxplot(statsbase.zscore(train.Fare))
statsbase.zscore(train.Fare)

statsplots.scatter(statsbase.zscore(train.Fare))
statsplots.scatter(train.Fare)

train[:Fare] = statsbase.zscore(train.Fare)

statsplots.plot(statsplots.plot_function(glm.logistic, train.Survived))
#=2
Shows difference in people who paid more survived vs people who paid less
tended to not survive. 
=#

#People who paid more were more likely to survive
statsplots.boxplot(train.Survived, train.Fare)
statsplots.bar(train.Survived, train.Fare)

train = train[[:Survived, :Pclass, :Sex, :Age, :SibSp, :Parch, :Fare]]
train = dataframes.dropmissing(train, :Age)

dataframes.describe(train)
train.Sex = mllabelutils.convertlabel([0, 1], train.Sex) # zero = male, female = 1

# TODO analysis on what predictors are most relevent to predicting the response variable
logreg = glm.glm(
    glm.@formula(Survived ~ Pclass),
    train, 
    glm.Binomial(),
    glm.LogitLink()
)

logreg = glm.glm(
    glm.@formula(Survived ~ Fare),
    train, 
    glm.Binomial(),
    glm.LogitLink()
)


xGrid = 0:1:maximum(train[:Fare])
statsplots.plot(xGrid, glm.predict(logreg), seriestype = :line)

glm.predict(logreg)
dataframes.DataFrame(
    Class = train.Pclass,
    Prediction = glm.predict(logreg)
)

hcat([1, 2], [3, 4])

#=
Make predictions: Use the intercept (first coeefficienct) 
with the second coefficient (for Pclass)
=#
firstclass, secondclass, thirdclass = 1, 2, 3
flux.sigmoid(glm.coef(logreg)[1] + glm.coef(logreg)[2] * firstclass)
flux.sigmoid(glm.coef(logreg)[1] + glm.coef(logreg)[2] * secondclass)
flux.sigmoid(glm.coef(logreg)[1] + glm.coef(logreg)[2] * thirdclass)
GLM.predict(logreg, dataframes.DataFrame(Survived = [1], Pclass = [3]))
glm.predict(logreg)

pred(x) = 1/(1+exp(-(glm.coef(logreg)[1] + glm.coef(logreg)[2]*x)))

xGrid = 0:0.1:maximum(train[:Pclass])

statsplots.plot(xGrid, pred.(xGrid))

mlogreg = glm.glm(
    glm.@formula(Survived ~ Pclass + Sex + Age + Parch + Fare), 
    train, 
    glm.Binomial(),
    glm.LogitLink()
)


statsplots.plot(statsplots.plot_function(glm.logistic, train.Survived))


menarche = CSV.File("/Users/greade01/Data/menarche.csv") |> DataFrames.DataFrame
menarche[:Total] = convert(Array{Int, 1}, menarche[:Total])
menarche[:MenarcheTotal] = menarche[:Menarche] ./ menarche[:Total]

StatsPlots.plot(menarche[:Age], menarche[:MenarcheTotal])


DataFramesMeta.@where(train, :Name, occursin.("Mr"))
DataFramesMeta.@select(train, filter(name -> occursin(String(name), "Mr")))
DataFramesMeta.@select(train, filter(name -> occursin(String(name), "Month"), names(train)))

DataFramesMeta.@select(train, [occursin("Name", String(name)) for name in names(train)])

train[occursin.("Mr.", train[:Name]), :]

mrmask = occursin.("Mr.", train[:Name])
train[:Name][mrmask]

DataFrames.by(train, :Title, DataFrames.nrow)


function title(names::Array{String, 1})
    acc::Array{String} = []
    for name in names
        occursin("Mr.", name) && push!(acc, "Mr")
        occursin("Mrs.", name) && push!(acc, "Mrs")
        occursin("Miss.", name) && push!(acc, "Miss")
        occursin("Master.", name) && push!(acc, "Master")
        occursin("Rev.", name) && push!(acc, "Rev")
        push!(acc, "Unknown")
    end
    acc
end

MLLabelUtils.labelfreq(train[:Title])

MLLabelUtils.label(
    MLLabelUtils.labelenc(train[:Title])
)

Flux.onehot("Mr", train[:Title])

# Pertial
g = Base.Fix1((x, y) -> x + y, 1)


occursin("m", skipmissing(["m", missing]))

skipmissing(missing) |> collect
