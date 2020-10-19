function [train,test]=data_to_crossvalind(indices,indicess,pos,neg,g,data)

    train=javaObject('weka.core.Instances',data,0);
    test=javaObject('weka.core.Instances',data,0);
    
    a=g;
    test_pos=(indices==a);
    train_pos=~test_pos;

     train_pos_biao=find(train_pos==1);
     test_pos_biao=find(test_pos==1);
     m=size(train_pos_biao,1);
     f=size(test_pos_biao,1);
 
    for d1=1:f
       r1=test_pos_biao(d1, 1);
       test.add(pos.instance(r1-1));
    end  
     for d2=1:m
       r2=train_pos_biao(d2, 1);
       train.add(pos.instance(r2-1));
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
       test.add(neg.instance(r3-1));
    end  
     for e2=1:s
       r4=train_neg_biao(e2, 1);
       train.add(neg.instance(r4-1));
     end     