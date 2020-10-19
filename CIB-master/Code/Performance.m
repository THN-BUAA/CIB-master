function [ PD,PF,Precision, F1,AUC,Accuracy,G_measure,MCC, Balance] = Performance( actual_label,predict_label, probPos)
%PERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) actual_label - The actual label, a column vetor, each row is an instance's class label.
%   (2) predict_label - The predicted label, a column vetor, each row is an instance label.
%   (3) probPos - The probability of being predicted as postive class.
% OUTPUTS:
%   PF,PF,..,MCC - A total of eight performance measures.

if numel(unique(actual_label)) < 2
    % error('The Input ''actual_label'' must have at least two different kinds of values.'); 
    [ PD,PF,Precision, F1,AUC,Accuracy,G_measure,MCC ] = deal(nan);
    return; % End program
end

if length(actual_label)~=length(predict_label)
    error('The dimensions of actual labels and predicted labels are not identitical.');
elseif ~exist('probPos') 
    error('The input ''probPos'' is empty!');
end


cf=confusionmat(actual_label,predict_label);
TP=cf(2,2);
TN=cf(1,1);
FP=cf(1,2);
FN=cf(2,1);

Accuracy = (TP+TN)/(FP+FN+TP+TN);
PD=TP/(TP+FN);
PF=FP/(FP+TN);
Precision=TP/(TP+FP);
F1=2*Precision*PD/(Precision+PD);
[X,Y,T,AUC]=perfcurve(actual_label, probPos, '1');% 
G_measure = (2*PD*(1-PF))/(PD+1-PF);
MCC = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));

Balance = 1 - sqrt(((0-PF)^2+(1-PD)^2)/2);

% if isnan(Precision)
%     Precision = 0;
% end
% if isnan(F1)
%     F1 = 0;
% end
% if isnan(MCC)
%     MCC = 0;
% end

end

