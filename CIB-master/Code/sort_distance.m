function [ disc,sort_disc,sort_index,sort_classvalue ] = sort_distance(original_data,synthetic_data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
syn_num=synthetic_data.numInstances();
orig_num=original_data.numInstances();
attrib_num=original_data.numAttributes();
disc=zeros(syn_num,orig_num);
sort_index=zeros(syn_num,orig_num);
sort_disc=zeros(syn_num,orig_num);
sort_classvalue=zeros(syn_num,orig_num);
for i=0:syn_num-1
    syn_a=synthetic_data.instance(i).toDoubleArray();
    syn_a=syn_a(1:attrib_num-1)';
    for j=0:orig_num-1
        orig=original_data.instance(j).toDoubleArray();
        orig=orig(1:attrib_num-1)';
        disc(i+1,j+1)=pdist2(syn_a,orig, 'euclidean');
    end
end


for i=1:syn_num
    [Y,I]=sort(disc(i,:));
    sort_index(i,:)=I;
    sort_disc(i,:)=Y;
    for j=1:orig_num
        sort_classvalue(i,j)=original_data.instance(I(j)-1).classValue();
    end
end



end

