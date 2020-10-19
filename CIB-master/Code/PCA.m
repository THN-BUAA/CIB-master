% function PCA(traindata)
%% data of cell transfer matrix
% load('train.mat', 'train')
% class(train);
% matix=cell2mat(train);
% matix=[];
clc;
clear all;
close all;

datafile='F:\Software\data\D\KC3.csv';
% class(datafile);
% [rows cols] = size(data);
% data=csvread(datafile);
data=xlsread(datafile);
label = data(:,end);

% yy_num=size(label);
% Y="Y";
% N="N";
% for xx=1:yy_num
%   if label==1
%       data(xx,end)=N;
%   else
%       data(xx,end)=Y;
%   end
%     
% end
weka2matlab
% Extracting positive data points
idx = (label==N);
pos_data = data(idx,:); 
pos_num = size(pos_data,1);

% Extracting negative data points
neg_data = data(~idx,:);
neg_num = size(neg_data,1);

indices = crossvalind('Kfold',pos_num,5);
indicess = crossvalind('Kfold',neg_num,5);


 for g=1:5  
% %   交叉验证的分组  
     train=[];
     test=[];
     
    a=g;
    test_pos=(indices==a);
    train_pos=~test_pos;

     train_pos_biao=find(train_pos==1);
     test_pos_biao=find(test_pos==1);
     m=size(train_pos_biao,1);
     f=size(test_pos_biao,1);
 
    for d1=1:f
       r1=test_pos_biao(d1, 1);
       test(d1,:)=pos_data (r1,:);
    end  
     for d2=1:m
       r2=train_pos_biao(d2, 1);
        train(d2,:)=pos_data (r2,:);
     end 
   
    b=g;
    test_neg=(indicess==b);
    train_neg=~test_neg;

     train_neg_biao=find(train_neg==1);
     test_neg_biao=find(test_neg==1);
     s=size(train_neg_biao,1);
     z=size(test_neg_biao,1);
 
    for e1=1:z
       r3=test_neg_biao(e1, 1);
       test(e1+d1,:)=neg_data (r3,:);
    end  
     for e2=1:s
       r4=train_neg_biao(e2, 1);
       train(e2+d2,:)=neg_data (r4,:);
     end    
     %% 主成分分析
     
    
     
     
     
 end
