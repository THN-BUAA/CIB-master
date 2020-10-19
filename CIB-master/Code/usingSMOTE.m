function [synthetic,synonly] = usingSMOTE( traindata,P,g)
% 已设置随机种子
% Minority Oversampling TEchnique (SMOTE)
%synthetic is the generated data mixed with the traindata
%  smalldata is the  minority data in traindata
%synonly is the synthetic data only
%smallsyn is the minority data of synthetic  
%options[1] -S num
 % Specifies the random number seed
 %(default 1)
 %options[2] -P percentage
 %Specifies percentage of SMOTE instances(minority data) to create
 %(default 100.0)
 %options[3] -K nearest-neighbors
 % Specifies the number of nearest neighbors to use.
 %(default 5)
 %options[4] -C value-index
 %Specifies the index of the nominal class value to SMOTE
 %(default 0: auto-detect non-empty minority class))
import weka.filters.Filter;
import weka.filters.*;
import weka.filters.supervised.instance.SMOTE.*;
import weka.core.converters.ArffLoader;
import weka.core.Instances;
import weka.core.Instance;
% import weka.filters.unsupervised.attribute.Add;
import java.lang.Math;


%javaaddpath('C:\Users\duying\Desktop\SMOTEBoost\SMOTEBoost\SMOTE.jar');


%synthetic=javaObject('weka.core.Instances',traindata,0);
%F=javaObject('weka.filters.Filter');
%abstract the minority data and its' number from traindata
 smallnum1=0;
 

for i=0:traindata.numInstances()-1
    if(traindata.instance(i).classValue()==0)
        smallnum1=smallnum1+1;
    end
end

synonly=javaObject('weka.core.Instances',traindata,0);
synthetic=javaObject('weka.core.Instances',traindata,0);
if(smallnum1==0)
    return;
end
S=javaObject('weka.filters.supervised.instance.SMOTE');
S.setPercentage(P);
S.setNearestNeighbors(5);
S.setInputFormat(traindata);
S.setRandomSeed(g);
% S.setRandomSeed(floor(Math.random()*100));
S.setClassValue('1'); % 假设标签属性取值为{Y,N}，1表示对标签为Y（第一个标签）的样本进行抽样，2表示对标签为N的样本进行抽样

% synthetic=weka.filter.Filter.useFilter(traindata, S);
% synthetic=Filter.useFilter(traindata, S);

% 



% javaaddpath('C:\Users\Administrator\Desktop\weka.jar');
% filter = javaObject('weka.filters.unsupervised.attribute.RenameNominalValues'); % RenameNominalValues
% filter.setOptions(weka.core.Utils.splitOptions(['-R ',num2str(insts.numAttributes()), '-N', 'Y:No, N:Yes']));
% 
% % filter.setAttributeIndex("last");
% % filter.setNominalLabels("N,Y");
% filter.setInputFormat(traindata);
% traindata1 = Filter.useFilter(traindata, filter);


try 
    synthetic=Filter.useFilter(traindata, S);
catch
%     synthetic = traindata;
    [mat, featureNames,targetNDX, stringVals, relationName] = weka2matlab(traindata, []);
    
    if contains(char(traindata.attribute(traindata.numAttributes()-1)), '{')
        temp = stringVals{end};
        if strcmpi(strtrim(temp{1}), 'buggy') || strcmpi(strtrim(temp{1}), 'YES') || strcmpi(strtrim(temp{1}), 'Y') || strcmpi(strtrim(temp{1}), 'faulty') || strcmpi(strtrim(temp{1}), 'defective') || strcmpi(strtrim(temp{1}), 'defect')...
                ||strcmpi(strtrim(temp{1}), 'fault')|| strcmpi(strtrim(temp{1}), 'ture')
            mat(mat(:,end)==0,end)=-1;
            mat(mat(:,end)==1,end)=0;  % 0表示无缺陷样本
            mat(mat(:,end)==-1,end)=1; % 1表示有缺陷样本
        end
    end
    
    minoSam = mat(mat(:,end)~=0,:);
    synSample = SMOTE( minoSam, size(minoSam,1), P); % self-defined, it cannot be same with WEKA_SMOTE owing to the randomness in the algorithm
    balData = [mat; synSample];
    
    labels = cell(size(balData,1),1);
    for i=1:size(balData,1)
        if balData(i,end)==1
            labels{i} = 'Yes';
        else
            labels{i} = 'No';
        end
    end
    synthetic = matlab2weka(relationName, featureNames, [num2cell(balData(:,1:end-1)), labels], [], 'descend');
    synonly = [];
%     filter = javaObject('weka.filters.unsupervised.attribute.Remove'); % 差一个大小写都不行
%     filter.setOptions(weka.core.Utils.splitOptions(['-R ', num2str(traindata.numAttributes())])); % 参数后必须有一个空格，即['-R ', num2str(traindata.numAttributes())]是不行的
%     filter.setInputFormat(wekaARFF);
%     wekaARFF1 = Filter.useFilter(wekaARFF, filter);
%     
%     filter = javaObject('weka.filters.unsupervised.attribute.Add'); % 差一个大小写都不行
%     filter.setOptions(weka.core.Utils.splitOptions({'-T NOM -N Defective -L Yes,No'}));
%     filter.setInputFormat(wekaARFF1);
%     wekaARFF = Filter.useFilter(wekaARFF1, filter);
%     
%     traindata1.instance(0).setValue(traindata1.numAttributes()-1, 'Y');
    
end
% S.getClassValue()

% disp('2');
% for i=traindata.numInstances():(synthetic.numInstances()-1)
%         synonly.add(synthetic.instance(i));
% end

% traindata.numInstances()
% % synthetic.numInstances()
% smallnum1
% synonly.numInstances()
% synonly.instance(0).classValue()

end

