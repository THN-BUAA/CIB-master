function [min_num_within_n,sort_index]=density_new(synthetic_data,original_data,n)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
syn_num=synthetic_data.numInstances();
orig_num=original_data.numInstances();
attrib_num=synthetic_data.numAttributes();
disc=zeros(syn_num,orig_num);
sort_index=zeros(syn_num,orig_num);
min_num_within_n=zeros(1,syn_num);
maj_num_within_n=zeros(1,syn_num);
for i=0:syn_num-1
        a=synthetic_data.instance(i).toDoubleArray();
        b=a(1:attrib_num-1)';
    for j=0:orig_num-1
        c=original_data.instance(j).toDoubleArray();
        d=c(1:attrib_num-1)';
        disc(i+1,j+1)=pdist2(d,b, 'euclidean'); 
    end
end

for i=0:syn_num-1
    [Y,I]=sort(disc(i+1,:));
    sort_index(i+1,:)=I;
    min_num=0;
    maj_num=0;
    k0=0;
    for j=1:orig_num      
        k0=k0+1;
        if(k0<n+1)
        if(original_data.instance(I(j)-1).classValue()==0)
            min_num=min_num+1;
        end

        end            
    end
    min_num_within_n(i+1)=min_num;  

end

