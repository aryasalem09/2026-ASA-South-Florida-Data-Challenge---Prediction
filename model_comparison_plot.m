clear; clc; rng(7);

train = readtable("train.csv");
outcomeName = "LBDHDD_outcome";

y = train.(outcomeName);
Xtrain = removevars(train, outcomeName);

tBest = templateTree('MaxNumSplits', 50);

mdlBoost = fitrensemble(Xtrain, y, 'Method', 'LSboost', 'Learners', tBest, 'numLearningCycles', 400, 'LearnRate', 0.05);
cvMdl = crossval(mdlBoost, 'KFold', 5);
lossBoost = sqrt(kfoldLoss(cvMdl, 'Mode', 'cumulative'));

tRF = templateTree('MaxNumSplits', 100);

mdlRF = fitrensemble(Xtrain, y, 'Method', 'Bag', 'Learners', tRF, 'numLearningCycles', 300);
cvRF = crossval(mdlRF, 'KFold', 5);
lossRF = sqrt(kfoldLoss(cvRF, 'Mode', 'cumulative'));

figure;
plot(lossBoost, 'LineWidth', 2);
hold on;
plot(lossRF, 'LineWidth', 2)
hold off;

xlabel('Number of Trees')
ylabel('Cross-Validated MSE')
title('Boosting vs Random Forest: Error Decay Comparison')
legend('Boosting', 'Random Forest', 'Location', 'northeast')
grid on

set(gca, 'FontSize', 11)
exportgraphics(gcf, 'model_strategy_comparison.png', 'Resolution', 300);

