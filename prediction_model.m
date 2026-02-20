clear; clc; rng(7);

train = readtable("train.csv");
test  = readtable("test.csv");

outcomeName = "LBDHDD_outcome";

y = train.(outcomeName);
Xtrain = removevars(train, outcomeName);
Xtest  = test;

tBest = templateTree('MaxNumSplits', 50);

finalMdl = fitrensemble(Xtrain, y, ...
    'Method','LSBoost', ...
    'Learners', tBest, ...
    'NumLearningCycles', 400, ...
    'LearnRate', 0.05);

pred = predict(finalMdl, Xtest);

sub = table(pred, 'VariableNames', {'pred'});
writetable(sub, "pred.csv");

save("finalModel.mat", "finalMdl")
%model in case we decide to use? probably wont be necessary..

disp("Done running, saved in pred.csv");