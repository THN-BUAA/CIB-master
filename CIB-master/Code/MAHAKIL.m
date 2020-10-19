function balancedData = MAHAKIL(Data,P)
%MAHAKIL 此处显示有关此函数的摘要
%   此处显示详细说明
% INPUTS:
%   (1) Data - a n*(d+1) array, n and d are the number of intstances and feature, respectively, The last column is the label {0,1} where 1 denotes defectproneness. 
%   (2) P    - a real number (0.3, 0.4, or 0.5), expected defect ratio after resampling.
% OUTPUT:
%   balanccedData - 
% Reference: [1] K. E. Bennin, J. Keung, P. Phannachitta, A. Monden and S. Mensah, "MAHAKIL: Diversity Based Oversampling Approach 
%            to Alleviate the Class Imbalance Issue in Software Defect Prediction," in IEEE Transactions on Software Engineering, 
%            vol. 44, no. 6, pp. 534-550, 1 June 2018.
%

if nargin ==1
    P = 0.5;
end

Dmin = Data(Data(:,end)==1,:);
k = size(Dmin, 1);

if (k/size(Data, 1)) >= P
    balancedData = Data;
    return;
end

T = floor((size(Data, 1)*P-k) / (1-P)); % the number of synthetic minority instances: (T+k)/(n+T)=P 

D = [];
try
     D = mahal(Dmin(:,1:end-1), Dmin(:,1:end-1));  
catch
     D =   eulDist(Dmin(:,1:end-1));
end

[temp, idx] = sort(D, 'descend');
Dmindist = Dmin(idx,:);

if rem(k, 2)==0 % exact division
    Nbin1 = Dmindist(1:k/2,:);
    Nbin2 = Dmindist((k/2+1):end,:);
else
    Nbin1 = Dmindist(1:floor(k/2),:);
    Nbin2 = Dmindist(floor(k/2)+1:end-1,:);
end

Nbin1X = Nbin1(:,1:end-1);
Nbin2X = Nbin2(:,1:end-1);

% Calculate split depth
depth = 1;
while size(Dmin,1)*(2^depth-1)/2 < T
    depth = depth + 1;
end

Xnew = SynInstance(Nbin1X, Nbin2X, depth);
balancedData = [Data; [Xnew, ones(size(Xnew,1),1)]];

end



function D = eulDist(dataX)
% FUNTIION: calculate the euclidean distance between each instance and sample mean
% INPUT:
%   dataX - a n*d array
% OUTPUT:
%   D - a n*1 vector, each element is the euclidean distance between each instance and sample mean 
%

n = size(dataX, 1);
D = zeros(n,1);
for i=1:n
    D(i) = sqrt(sum((dataX(i,:)-mean(dataX)).^2));
end

end



function Xnew = SynInstance(Nbin1X, Nbin2X, depth, i)
% FUNTION: Generate synthetic minority class instances according to K. E. Bennin et.al., paper [1].
% INPUTS:
%   (1) Nbin1X - a n1*d matrix, d denotes the number of features;
%   (2) Nbin2X - a n1*d matrix, d denotes the number of features;
%   (3) depth  - an integer denotes the split depth
%   (4) i      - current depth
% OUTPUTS:
%
%
if nargin == 3
    i = 1;
end

Xnew = (Nbin1X+Nbin2X)/2; % generate synthetic instances

Xnew1 = [];
Xnew2 = [];
if i < depth
    Xnew1 = SynInstance(Xnew, Nbin1X, depth, i+1);
    Xnew2 = SynInstance(Xnew, Nbin2X, depth, i+1);
end

Xnew = [Xnew; Xnew1; Xnew2];
end
