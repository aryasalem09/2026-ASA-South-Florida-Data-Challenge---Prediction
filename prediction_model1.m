%this was here to test all the ways I could predict and the RSME score...

clear; clc; rng(7);


train  = readtable("train.csv");
test   = readtable("test.csv");


outcomeName = "LBDHDD_outcome";

y = train.(outcomeName);
Xtrain = removevars(train, outcomeName);
Xtest  = test;

y = y(:); 


Xtrain = convertvars(Xtrain, @isstring, 'categorical');
Xtest  = convertvars(Xtest,  @isstring, 'categorical');

isCharTrain = varfun(@ischar, Xtrain, 'OutputFormat','uniform');
charVars = Xtrain.Properties.VariableNames(isCharTrain);
for v = charVars
    Xtrain.(v{1}) = categorical(string(Xtrain.(v{1})));
    Xtest.(v{1})  = categorical(string(Xtest.(v{1})));
end


numMask = varfun(@isnumeric, Xtrain, 'OutputFormat','uniform');
numVars = Xtrain.Properties.VariableNames(numMask);
for v = numVars
    medv = median(Xtrain.(v{1}), 'omitnan');
    Xtrain.(v{1}) = fillmissing(Xtrain.(v{1}), 'constant', medv);
    Xtest.(v{1})  = fillmissing(Xtest.(v{1}),  'constant', medv);
end


catMask = varfun(@iscategorical, Xtrain, 'OutputFormat','uniform');
catVars = Xtrain.Properties.VariableNames(catMask);
for v = catVars
    Xtrain.(v{1}) = addcats(Xtrain.(v{1}), "Unknown");
    Xtest.(v{1})  = addcats(Xtest.(v{1}),  "Unknown");
    Xtrain.(v{1}) = fillmissing(Xtrain.(v{1}), 'constant', "Unknown");
    Xtest.(v{1})  = fillmissing(Xtest.(v{1}),  'constant', "Unknown");
end


baselineRMSE = sqrt(mean((y - mean(y)).^2));
fprintf("Baseline (predict mean) RMSE: %.4f | std(y): %.4f\n\n", baselineRMSE, std(y));


cvRMSE = @(ytrue, ypred) sqrt(mean((ytrue(:) - ypred(:)).^2));

boostSplits = [20 50 100];         
boostLR     = [0.03 0.05 0.1];      
boostCycles = [400 800 1200];      

bestBoostRMSE = inf;
bestBoostParams = [];

fprintf("=== Tuning Boosting (LSBoost) ===\n");
for ms = boostSplits
    for lr = boostLR
        for nc = boostCycles
            t = templateTree('MaxNumSplits', ms);

            mdlCV = fitrensemble(Xtrain, y, ...
                'Method','LSBoost', ...
                'Learners', t, ...
                'NumLearningCycles', nc, ...
                'LearnRate', lr, ...
                'KFold', 5);

            yhat = kfoldPredict(mdlCV);
            rmse = cvRMSE(y, yhat);

            fprintf("Boost: MaxSplits=%d LR=%.3f Cycles=%d -> RMSE=%.4f\n", ms, lr, nc, rmse);

            if rmse < bestBoostRMSE
                bestBoostRMSE = rmse;
                bestBoostParams = [ms lr nc];
            end
        end
    end
end
fprintf("BEST BOOST: MaxSplits=%d LR=%.3f Cycles=%d RMSE=%.4f\n\n", ...
    bestBoostParams(1), bestBoostParams(2), bestBoostParams(3), bestBoostRMSE);

%no errors up til this point, I think

rfSplits = [20 50 100];
rfTrees  = [300 600 1000];

bestRFRMSE = inf;
bestRFParams = [];

fprintf("=== Tuning Random Forest (Bagging) ===\n");
for ms = rfSplits
    for nt = rfTrees
        t = templateTree('MaxNumSplits', ms);

        mdlCV = fitrensemble(Xtrain, y, ...
            'Method','Bag', ...
            'Learners', t, ...
            'NumLearningCycles', nt, ...
            'KFold', 5);

        yhat = kfoldPredict(mdlCV);
        rmse = cvRMSE(y, yhat);

        fprintf("RF: MaxSplits=%d Trees=%d -> RMSE=%.4f\n", ms, nt, rmse);

        if rmse < bestRFRMSE
            bestRFRMSE = rmse;
            bestRFParams = [ms nt];
        end
    end
end
fprintf("BEST RF: MaxSplits=%d Trees=%d RMSE=%.4f\n\n", bestRFParams(1), bestRFParams(2), bestRFRMSE);


useRF = bestRFRMSE < bestBoostRMSE;

if useRF
    fprintf("WINNER: Random Forest (Bagging)\n");
    ms = bestRFParams(1);
    nt = bestRFParams(2);

    tFinal = templateTree('MaxNumSplits', ms);
    finalMdl = fitrensemble(Xtrain, y, ...
        'Method','Bag', ...
        'Learners', tFinal, ...
        'NumLearningCycles', nt);
else
    fprintf("WINNER: Boosting (LSBoost)\n");
    ms = bestBoostParams(1);
    lr = bestBoostParams(2);
    nc = bestBoostParams(3);

    tFinal = templateTree('MaxNumSplits', ms);
    finalMdl = fitrensemble(Xtrain, y, ...
        'Method','LSBoost', ...
        'Learners', tFinal, ...
        'NumLearningCycles', nc, ...
        'LearnRate', lr);
end


imp = predictorImportance(finalMdl);
[sortedImp, idx] = sort(imp, 'descend');

topK = 10;
topNames = Xtrain.Properties.VariableNames(idx(1:topK));
topVals  = sortedImp(1:topK);

importanceTable = table(topNames', topVals', 'VariableNames', {'Predictor','Importance'});
disp(importanceTable);


pred = predict(finalMdl, Xtest);
sub = table(pred, 'VariableNames', {'pred'});
writetable(sub, "pred.csv");
disp("FINAL submission file created: pred.csv");


save("finalModel.mat", "finalMdl", "bestBoostParams", "bestBoostRMSE", "bestRFParams", "bestRFRMSE", "importanceTable");
disp("Saved: finalModel.mat");