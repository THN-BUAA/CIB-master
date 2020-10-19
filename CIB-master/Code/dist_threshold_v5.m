function [dth,d1] =dist_threshold_v5( original_data,synthetic_data,nth )


syn_num=synthetic_data.numInstances();
orig_num=original_data.numInstances();
attrib_num=synthetic_data.numAttributes();
dmin=zeros(1,syn_num);
d1=zeros(syn_num,orig_num);
%calculate dth
for i=0:syn_num-1
    a=synthetic_data.instance(i).toDoubleArray();
    b=a(1:attrib_num-1)';   
    for j=0:orig_num-1        
        c=original_data.instance(j).toDoubleArray();
        d=c(1:attrib_num-1)';
        d1(i+1,j+1)=pdist2(d,b, 'euclidean');       
    end
    sum=0;
    d0=d1(i+1,:);
    d0=sort(d0);
for k=1:nth
    sum=sum+d0(k);
end
dmin(i+1)=sum;    
end
[Y,I]=min(dmin);
dmin_ascend=sort(d1(I,:));
dth=dmin_ascend(nth);



end

