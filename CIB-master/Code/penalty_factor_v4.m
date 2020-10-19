function [bi,sort_index] = penalty_factor_v4(  data,syn_index,nth ,t1,t2,k)
import weka.core.Instances;
import weka.core.Instance;
original_data=javaObject('weka.core.Instances',data,0);
original_min=javaObject('weka.core.Instances',data,0);
original_maj=javaObject('weka.core.Instances',data,0);

p1=4;
p2=2;

syms k1 
K1=solve(exp(-k1*((k-1)/k).^p1)==t1);%反函数，将少数类的密度映射到[0.9,1],指数越大，越往上凸
syms k2 
K2=solve(1-exp(-k2*((k-1)/k).^p2)==t2);%正函数，将多数类的密度[0,(k-1)/k]映射到[t3,t2]，指数越大，越往下凹



for i=0:syn_index-1
    original_data.add(data.instance(i));%abstract the original data
    if(data.instance(i).classValue()==0)
        original_min.add(data.instance(i));% abstract the original minority data
    else
        original_maj.add(data.instance(i));%abstract the original manority data
    end
end
synthetic_data=javaObject('weka.core.Instances',data,0);

for i=syn_index:data.numInstances()-1
    synthetic_data.add(data.instance(i));% abstract the synthetic data   
end
bi=zeros(1,synthetic_data.numInstances());
[dth,d]=dist_threshold_v5(original_data,synthetic_data,nth);
[numbers_of_maj_within_dth,numbers_of_min_within_dth,sort_index,min_num_within_k]=density_v51(d,original_data,dth,k);

for i=1:synthetic_data.numInstances()
    if(numbers_of_min_within_dth(i)<=5)
    if numbers_of_min_within_dth(i)>0&&numbers_of_maj_within_dth(i)<=10
        x=(5-numbers_of_min_within_dth(i))/5;
        bi(i)=exp(-K1*x.^p1);
    end
     if numbers_of_min_within_dth(i)==0&&numbers_of_maj_within_dth(i)<=10
            x=(10-numbers_of_maj_within_dth(i))/10;
            bi(i)=1-exp(-K2*(x).^p2);
     end
     if numbers_of_maj_within_dth(i)>10
                bi(i)=0.1;
     end
     k=max(min_num_within_k);
    if(numbers_of_min_within_dth(i)==0&&numbers_of_maj_within_dth(i)==0)
        if(min_num_within_k(i)>0)
            x=(k-min_num_within_k(i))/k;
            bi(i)=exp(-K1*x.^p1);
        else
            bi(i)=0.1;
        end
    end
    else 
         bi(i)=1.0;
    end
end
% numbers_of_maj_within_dth,numbers_of_min_within_dth,min_num_within_k,bi 
    
end

