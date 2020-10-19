function [ train, test ] = sample(data)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
import weka.core.Instances;
import weka.core.Instance;
pos=javaObject('weka.core.Instances',data,0);
train=javaObject('weka.core.Instances',data,0);
test=javaObject('weka.core.Instances',data,0);
neg=javaObject('weka.core.Instances',data,0);

%indices = crossvalind('Kfold',traindata.setClassIndex,5);
for i=0:data.numInstances()-1
    if(data.instance(i).classValue()==1)
        neg.add(data.instance(i));
    else
        pos.add(data.instance(i));
    end
end
pos_num=pos.numInstances();
neg_num=neg.numInstances();
pos_num_two=round(pos_num*0.2);
neg_num_two=round(neg_num*0.2);



% for i=1:pos_num_eight
%     r=round(rand(1)*10);
%     while((ismember(r,temp)==1)||(r>pos_num_eight))
%     r=round(rand(1)*10);
%     end
%     temp(i+1)=r;
%     train.add(pos.instance(r));
% end
% temp
% temp=10000000*ones(1,neg_num_eight);
% 
% for i=1:neg_num_eight
%      r=round(rand(1)*10);
%     while((ismember(r,temp)==1)||(r>neg_num_eight))
%     r=round(rand(1)*10);
%     end
%     temp(i+1)=r;
%      train.add(neg.instance(r));
% end
% 
% temp
temp_pos=10000000*ones(1,pos_num_two);

for i=0:pos_num_two-1
     r=round(randi(pos_num,1));
    while((ismember(r-1,temp_pos)==1)||(r>pos_num))
    r=round(randi(pos_num,1));
    end
    temp_pos(i+1)=r-1;
    test.add(pos.instance(r-1));
end

temp_neg=10000000*ones(1,neg_num_two);

for j=0:neg_num_two-1
     r1=round(randi(neg_num,1));
    
    while((ismember(r1-1,temp_neg)==1)||(r1>neg_num))
    r1=round(randi(neg_num,1));
   
    end
    temp_neg(j+1)=r1-1;
     test.add(neg.instance(r1-1));
end

for k=0:neg.numInstances()-1
    if(ismember(k,temp_neg)~=1)
        train.add(neg.instance(k));
    end
end

for k=0:pos.numInstances()-1
    if(ismember(k,temp_pos)~=1)
        train.add(pos.instance(k));
    end
end
end

