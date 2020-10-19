function wekaOBJ = matlab2weka(name, featureNames, data,targetIndex, strOrder)
% Convert matlab data to a weka java Instances object for use by weka
% classes. 
%
% name           - A string, naming the data/relation
%
% featureNames   - A cell array of d strings, naming each feature/attribute
%
% data           - An n-by-d matrix with n, d-featured examples or a cell
%                  array of the same dimensions if string values are
%                  present. You cannot mix numeric and string values within
%                  the same column. 
%
% wekaOBJ        - Returns a java object of type weka.core.Instances
%
% targetIndex    - The column index in data of the target/output feature.
%                  If not specified, the last column is used by default.
%                  Use the matlab convention of indexing from 1.
%
% Written by Matthew Dunham
    if ~exist('strOrder','var')||isempty(strOrder) % myself
        strOrder = 'ascend';
    end

    if(~wekaPathCheck),wekaOBJ = []; return,end
    if~exist('targetIndex','var')||isempty(targetIndex)
        targetIndex = numel(featureNames); %will compensate for 0-based indexing later
    end

    import weka.core.*;%������
    vec = FastVector();
    if(iscell(data))
        for i=1:numel(featureNames)
            if(ischar(data{1,i}))
                attvals = unique(data(:,i)); % ȥ���ظ�ֵ����������
                if numel(attvals)==2 && strcmp(strOrder,'descend') % ��������,% myself
                    temp = attvals{1};
                    attvals{1} = attvals{2};
                    attvals{2} = temp;
                end
                values = FastVector();
                for j=1:numel(attvals)
                   values.addElement(attvals{j});
                end
                vec.addElement(Attribute(featureNames{i},values));% Must be a string, can't be a char.
            else
                vec.addElement(Attribute(featureNames{i})); 
            end
        end 
    else
        for i=1:numel(featureNames)
            vec.addElement(Attribute(featureNames{i})); 
        end
    end
    wekaOBJ = Instances(name,vec,size(data,1));
    if(iscell(data))
        for i=1:size(data,1)
            
            try 
                inst = Instance(numel(featureNames));
            catch
                inst = DenseInstance(numel(featureNames));
            end
            for j=0:numel(featureNames)-1
               inst.setDataset(wekaOBJ);
               inst.setValue(j,data{i,j+1});
            end
            wekaOBJ.add(inst);
        end
    else
        for i=1:size(data,1)
            wekaOBJ.add(Instance(1,data(i,:)));
        end
    end
    wekaOBJ.setClassIndex(targetIndex-1);
end