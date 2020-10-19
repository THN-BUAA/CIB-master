function [ synData ] = SMOTE( data, T, N, k, seed)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) data - Minority class samples, a T*(d+1) matrix, d denotes the number of features, the last column is the label.
%   (2) T    - Number of minority class samples.
%   (3) N    - Amount of SMOTE N%, e.g., 100,200,300
%   (4) k    - Number of the nearest neighbors.
% OUTPUTS:
%   synData  - (N/100)* T synthetic minority class samples.
%
% Reference: N.V. Chawla, K.W. Bowyer, L.O. Hall, and W.P. Kegelmeyer, SMOTE:
% synthetic minority over-sampling technique, Journal of Artificial Intelligence 
% Research, vol.16, pp.321–357, 2002

if length(unique(data(:,end)))~=1
    error('Error: data must be the minority samples');
end

if ~exist('seed','var')||isempty(seed)
    seed = 1;
end
% rand('seed', seed);
rng(seed);

if N <100
    T = floor((N/100)*T);
    N = 100;
end

if ~exist('k','var')||isempty(k)
    k=5;
end


% shuffle the minority samples
idx = randperm(T,T); % Disturb the order of minority samples
data = data(idx,:);

N = floor(N/100);
numattrs = size(data,2) - 1;

synData = zeros(N*T,size(data,2)-1);
synData2 = zeros(N*T,size(data,2)-1);

dataX = data(:,1:end-1);

% Distance between instances
distM = dist(dataX');
distM = distM - eye(size(distM,1),size(distM,1)); % 对角线元素全部变为负数，后续好删除

% Index of neighbors of T instance
% idx = randperm(size(data,1),T); % Disturb the order of minority samples
neigIndex = zeros(T, k);    % Initialization 
for i=1:T % each minority sample
    [val, ord] = sort(distM(i,:)); % smallest->biggest for each row
    neigIndex(i,:) = ord(2:(k+1));
end

count = 1;
while N~=0
    for i = 1:T % each minority samples
        sample = dataX(i,:);
        nn = randperm(k,1);
        synData(count,:) = sample + rand(1,numattrs).*(dataX(neigIndex(i,nn),:)-sample);
%         temp = dataX(neigIndex(i,nn));
%         synData(count,:) = sample + rand(1,numattrs).*(temp-sample);
        count = count + 1;
    end
    N = N-1;
end

% % Index of neighbors of T instance
% idx = randperm(size(data,1),T); % Disturb the order of minority samples
% neigIndex = zeros(T, k);    % Initialization 
% for i=1:T
%     [val, ord] = sort(distM(idx(i),:)); % smallest->biggest for each row
%     neigIndex(i,:) = ord(2:(k+1));
% end
% 
% count= 1;
% while N~=0
%     for i = 1:T
%         sample = dataX(idx(i),:);
%         nn = randperm(k,1);
%         synData2(newindex,:) = sample + rand*(dataX(neigIndex(i,nn),:)-sample);
%         synData(newindex,:) = sample + rand(1,numattrs).*(dataX(neigIndex(i,nn))-sample);
%         count= count+ 1;
%     end
%     N = N-1;
% end

synData = [synData,ones(size(synData,1),1)];

end

