
clear;

% Add jar package
javaaddpath('.\weka.jar');
javaaddpath('.\WeightedSmoteBoost_v1.jar');%
javaaddpath('.\RAMOBoost.jar');      % 
javaaddpath('.\newEvaluation.jar'); %


dataNames= {'CM1.arff','KC1.arff','KC3.arff','MC1.arff','MC2.arff','MW1.arff','PC1.arff','PC2.arff','PC3.arff','PC4.arff','PC5.arff','JM1.arff'};
perfNames = {'PD','Precision','F1','G_mean','AUC', 'Accuracy','PF','G_measure','MCC','Balance'};
perfs = cell(1,numel(dataNames));
modelNames = {'CIB','MAHAKIL', 'AdaC2', 'AdaBoost', 'SMOTE', 'RUS', 'None'};
runs = 30;
folds = 5;

savePath = 'E:\Documents\Experiments\CIB\RQ1';
if ~exist(savePath,'dir')
    mkdir(savePath);
end
    
warning('off');
for d=1:numel(dataNames)
    disp(['Data:', num2str(d), '/', num2str(numel(dataNames))]);
    
    datafile=[filePath, dataNames{d}];
    
    data=loadARFF(datafile);
    data.setClassIndex(data.numAttributes()-1);
    p1=1.0;  p2=1.0;
    nth=40;
    t1=0.9;  t2=0.9;
    k=10;
    
    %
    import weka.core.Instances;
    import weka.core.Instance;
    pos=javaObject('weka.core.Instances',data,0);
    neg=javaObject('weka.core.Instances',data,0);
    
    
    for i=0:data.numInstances()-1 % Traverse each sample
        if(data.instance(i).classValue()==1)
            neg.add(data.instance(i)); % NOTE: 1 - negative (majority)
        else
            pos.add(data.instance(i));
        end
    end
    pos_num=pos.numInstances();
    neg_num=neg.numInstances();
    P=(floor(neg_num/pos_num)-1)*100; % 
    if P==0
        P = 100;
    end
%     P=300;
    cost=P/100+1;
    
    % ³õÊ¼»¯ 
    perfMAHAKIL=zeros(runs,numel(perfNames)); % 
    perfCIB=zeros(runs,numel(perfNames));
    perfsABS=zeros(runs,numel(perfNames));
    perfRAMOBoost=zeros(runs,numel(perfNames));
    perfAdaC2=zeros(runs,numel(perfNames));
    perfAdaBoost=zeros(runs,numel(perfNames));
    perfSMOTE = zeros(runs,numel(perfNames));
    perfNone = zeros(runs,numel(perfNames));
    perfRUS = zeros(runs,numel(perfNames));

    
    baseClassifier='weka.classifiers.trees.J48';
    classifier_options='-I 20 -W weka.classifiers.trees.J48'; 
    
    for i=1:runs
        disp(['runs:', num2str(i), '/', num2str(runs)]);
        recall=zeros(8,1);
        precision=zeros(8,1);
        f_measure=zeros(8,1);
        g_mean=zeros(8,1);
        auc=zeros(8,1);
        confusionMetrix=zeros(8,4);
        correctMetrix=zeros(8,4);
        Accuracy=zeros(8,1);
        PF=zeros(8,1);
        G_measure=zeros(8,1);
        MCC=zeros(8,1);
        Balance=zeros(8,1);
        K=false;
        
        rng(i);%rand('seed',0)
        indices = crossvalind('Kfold',pos_num,folds);
        indicess = crossvalind('Kfold',neg_num,folds);
        
        % Initialization
        a1 = zeros(folds, numel(perfNames)); a2 = zeros(folds, numel(perfNames));  a3 = zeros(folds, numel(perfNames)); a4 = zeros(folds, numel(perfNames));
        a5 = zeros(folds, numel(perfNames)); a6 = zeros(folds, numel(perfNames)); a7 = zeros(folds, numel(perfNames));
         
        for j=1:folds
            posCopy = pos;
            negCopy = neg;
            train=javaObject('weka.core.Instances',data,0);
            test=javaObject('weka.core.Instances',data,0);
            trainNeg=javaObject('weka.core.Instances',data,0);
            
            % 
            a=j;
            test_pos=(indices==a);
            train_pos=~test_pos;
            
            train_pos_idx=find(train_pos==1);
            test_pos_idx=find(test_pos==1);
            m=size(train_pos_idx,1);
            f=size(test_pos_idx,1);
            
            % postive samples for testing dataset
            for d1=1:f
                r1=test_pos_idx(d1, 1);
                test.add(pos.instance(r1-1));
            end
            
            % positive sampels for training dataset
            for d2=1:m
                r2=train_pos_idx(d2, 1);
                train.add(pos.instance(r2-1));
            end
            trainPos = train;
            
            %
            b=j;
            test_neg=(indicess==b);
            train_neg=~test_neg;
            
            train_neg_idx=find(train_neg==1);
            test_neg_idx=find(test_neg==1);
            s=size(train_neg_idx,1);
            z=size(test_neg_idx,1);
            
            % negative samples for testing dataset
            for e1=1:z
                r3=test_neg_idx(e1, 1);
                test.add(neg.instance(r3-1));
            end
            
            % negative samples for training dataset
            for e2=1:s
                r4=train_neg_idx(e2, 1);
                train.add(neg.instance(r4-1));
                trainNeg.add(neg.instance(r4-1));
            end
                
            
            %% CIB
            %     [ train, test ] = sample(traindata);
            [synthetic,synonly] = usingSMOTE( train,P,j); % 
            [bi,sort_index]=penalty_factor_v4( synthetic,train.numInstances(),nth,t1,t2,k);
            
            Utils_CIB = javaObject('weka.core.Utils');
            options_CIB = Utils_CIB.splitOptions(classifier_options);
            javaaddpath('WeightedSmoteBoost_v1.jar');%
            CIB_classifier = javaObject('weka12.WeightedSmoteBoost_v1');
            CIB_classifier.setOptions(options_CIB);
            try
                CIB_classifier.buildClassifier(synthetic,bi,train.numInstances(),sort_index,p1,p2,k);
                a1(j,:) = evaluation_weka_classifier( CIB_classifier,test );
            catch
                a1(j,:) = nan(1,numel(perfNames));
            end            
            
            %% MAHAKIL
            [mdata,featureNames,targetNDX,stringVals,relationName] =weka2matlab(train,[]); %×¢ÒâY->0, N->1
            mdata(mdata(:,end)==1,end)=2;
            mdata(mdata(:,end)==0,end)=1;
            mdata(mdata(:,end)==2,end)=0;            
            mdata = MAHAKIL(mdata);
            mdata(mdata(:,end)==1,end)=2;
            mdata(mdata(:,end)==0,end)=1;
            mdata(mdata(:,end)==2,end)=0;
            
            label = cell(size(mdata,1),1);
            temp = mdata(:,end);
            for j0=1:size(mdata,1)
                if (temp(j0)==0) % 
                    label{j0} = 'defetive';
                else
                    label{j0} = 'nonedefective';
                end
            end           
            wekaOBJ = matlab2weka(relationName, featureNames, [num2cell(mdata(:,1:end-1)), label]);   
            Utils = javaObject('weka.core.Utils');
            MAHAKIL_classifier = javaObject(baseClassifier);
            try
                MAHAKIL_classifier.buildClassifier(wekaOBJ);
                a2(j,:) = evaluation_weka_classifier( MAHAKIL_classifier,test );
            catch
                a2(j,:) = nan(1,numel(perfNames));
            end
                       
            
            %% AdaC2
            Utils_AdaC2 = javaObject('weka.core.Utils');
            options_AdaC2 = Utils_AdaC2.splitOptions(classifier_options);
            javaaddpath('AdaC2.jar');%
            AdaC2=javaObject('weka10.AdaC2');
            AdaC2.setOptions(options_AdaC2);
            try
                AdaC2.buildClassifier(train,cost);
                a3(j,:) = evaluation_weka_classifier( AdaC2,test );
            catch
                a3(j,:) = nan(1,numel(perfNames));
            end
            
             %% AdaBoost
            Utils_AdaBoost = javaObject('weka.core.Utils');
            options_AdaBoost = Utils_AdaBoost.splitOptions(classifier_options);
            AdaBoost = javaObject('weka.classifiers.meta.AdaBoostM1');
            AdaBoost.setOptions(options_AdaBoost);
            try
                AdaBoost.buildClassifier(train);
                a4(j,:) = evaluation_weka_classifier( AdaBoost,test );
            catch
                a4(j,:) = nan(1,numel(perfNames));
            end  
            
            %% SMOTE
            SMOTE_classifier = javaObject(baseClassifier); 
            try
                SMOTE_classifier.buildClassifier(synthetic);
                a5(j,:) = evaluation_weka_classifier( SMOTE_classifier,test );
            catch
                a5(j,:) = nan(1,numel(perfNames));
            end
            
            %% RUS
            % randomly select some majority class samples from training dataset
            RUSTrain = javaObject('weka.core.Instances',data,0);
            rng(0);
            selectedDefIdx = randperm(trainPos.numInstances,trainNeg.numInstances);
            for j0 = 1:numel(selectedDefIdx)
                RUSTrain.add(trainPos.instance(selectedDefIdx(j0)-1));
            end
            % combine selected majority class samples and minority class sampels in training dataset
            for j0 = 1:trainNeg.numInstances
                RUSTrain.add(trainNeg.instance(j0-1));
            end
            
            RUS_classifier = javaObject(baseClassifier);
            try
                RUS_classifier.buildClassifier(RUSTrain);
                a6(j,:) = evaluation_weka_classifier( RUS_classifier,test );
            catch
                a6(j,:) = nan(1,numel(perfNames));
            end
               
            
            %% None
            base_classifier = javaObject(baseClassifier);
            try
                base_classifier.buildClassifier(train);
                a7(j,:) = evaluation_weka_classifier( base_classifier,test );
            catch
                a7(j,:) = nan(1,numel(perfNames));
            end
            

        end % end folds
        perfCIB(i,:)=nanmean(a1); 
        perfMAHAKIL(i,:)=nanmean(a2);
        perfAdaC2(i,:)=nanmean(a3);
        perfAdaBoost(i,:)=nanmean(a4);
        perfSMOTE(i,:)=nanmean(a5);
        perfRUS(i,:) = nanmean(a6);
        perfNone(i,:)=nanmean(a7);
        
    end % end runs
    perfs{d} = {perfCIB, perfMAHAKIL, perfAdaC2, perfAdaBoost, perfSMOTE, perfRUS, perfNone};
    
    
    save([savePath,'\perfs.mat'],'perfs');  
end












