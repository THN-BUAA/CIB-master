function perf = evaluation_weka_classifier( classifier,test )
% function [ Recall,Precision,F_measure,G_mean,AUC ,confusionMetrix,correctMetrix,Accuracy,PF,G_measure,MCC,Balance] = evaluation_weka_classifier( classifier,test )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%javaaddpath('G:\Software\jar\myweka4\newEvaluation.jar');
javaaddpath('.\newEvaluation.jar');
myeval=javaObject('myweka4.NewEvaluation');
% myeval=javaObject('weka.classifiers.Evaluation');
eval=myeval.EvaluateModel(classifier, test);
m_correct=eval.correct();
m_incorrect=eval.incorrect();
per_correct=eval.pctCorrect();
per_incorrect=eval.pctIncorrect();
confusionMetrix=zeros(1,4);
correctMetrix=zeros(1,4);
correctMetrix(1,1)=m_correct;
correctMetrix(1,2)=m_incorrect;
correctMetrix(1,3)=per_correct;
correctMetrix(1,4)=per_incorrect;


TP=eval.numTruePositives(0);
FN=eval.numFalseNegatives(0);
FP=eval.numFalsePositives(0);
TN=eval.numTrueNegatives(0);


confusionMetrix(1,1)=TP;
confusionMetrix(1,2)=FN;
confusionMetrix(1,3)=FP;
confusionMetrix(1,4)=TN;

TN_rate=TN/(TN+FP);
Recall=TP/(TP+FN);%also TP_rate
if(TP+FP==0)
    Precision=0;
else
    Precision=TP/(TP+FP);
end
if(Recall+Precision==0)
    F_measure=0;
else
    F_measure=(2*Recall*Precision)/(Recall+Precision);
end
G_mean=sqrt(Recall*TN_rate);

Accuracy = (TP+TN)/(FP+FN+TP+TN);
% Recall=TP/(TP+FN);
PF=FP/(FP+TN);
% Precision=TP/(TP+FP);
% F1=2*Precision*Recall/(Precision+Recall);
% [X,Y,T,AUC]=perfcurve(actual_label, probPos, '1');%
G_measure = (2*Recall*(1-PF))/(Recall+1-PF);
if(TP+FP==0 || TP+FN==0 || TN+FP==0 || TN+FN==0)
    MCC=0;
else
MCC = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));
end

Balance = 1 - sqrt(((0-PF)^2+(1-Recall)^2)/2);

AUC=eval.areaUnderROC(0);

perf = zeros(1,10);
perf(1) = Recall;perf(2)=Precision;perf(3)=F_measure; perf(4)=G_mean; perf(5)=AUC; perf(6)=Accuracy;perf(7)=PF;perf(8)=G_measure;perf(9)=MCC;perf(10)=Balance;
end

