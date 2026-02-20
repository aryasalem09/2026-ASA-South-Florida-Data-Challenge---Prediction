clear; clc;

load("finalModel.mat");
imp = predictorImportance(finalMdl);
[sortedImp, idx] = sort(imp, 'descend');

topK = 10;
topVals = sortedImp(1:topK);
topNames = finalMdl.PredictorNames(idx(1:topK));

figure;
bar(topVals)
xticks(1:topK)
xticklabels(topNames)
xtickangle(45)
ylabel('Relative Importance')
title('Top 10 Predictor Importances to HDL')

set(gca, 'FontSize', 11)

exportgraphics(gcf, 'importance_plot.png', 'Resolution', 300);

%idk if we'll use this lets see