function [numbers_of_maj_within_dth,numbers_of_min_within_dth,sort_index,min_num_within_k]=density_v51(disc,original_data,dth,k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[syn_num,orig_num]=size(disc);

numbers_of_maj_within_dth=zeros(1,syn_num);
numbers_of_min_within_dth=zeros(1,syn_num);
min_num_within_k=zeros(1,syn_num);
sort_index=zeros(syn_num,orig_num);


for i=0:syn_num-1

    sum_maj=0;
    sum_min=0;
    for j=0:orig_num-1
        e=disc(i+1,j+1);
         if(e<=dth)          

            if (original_data.instance(j).classValue()==1)
                sum_maj=sum_maj+1;
            end
            if(original_data.instance(j).classValue()==0)
                sum_min=sum_min+1;
            end
         end
    end

   numbers_of_maj_within_dth(i+1)=sum_maj;
   numbers_of_min_within_dth(i+1)=sum_min;
   
   [Y,I]=sort(disc(i+1,:));
    sort_index(i+1,:)=I;
  
    min_num=0;
    k0=0;
    for j=1:orig_num
      
        k0=k0+1;
        if(k0<k+1)
        if(original_data.instance(I(j)-1).classValue()==0)
            min_num=min_num+1;
        end
        end            
    end
    min_num_within_k(i+1)=min_num;
end




end

