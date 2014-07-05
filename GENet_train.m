function [GENet,data]=GENet_train(GENet,data,label)
if size(data,1) ~= size(label,1)
    error('size(data,1) ~= size(label,1)')           
end

for stage = 1:size(GENet.layer,2)
    if strcmp(GENet.layer{stage}.type,'PCA')
        fprintf('the %d layer is PCA\n',stage);
        if ~isfield(GENet.layer{stage},'ReducedDim')
            error('GENet.layer{%d}.ReducedDim is not found!',stage)
        end
        options=[];
        options.ReducedDim=GENet.layer{stage}.ReducedDim;
        [eigvector,eigvalue] = PCA(data,options);
        GENet.layer{stage}.eigvector=eigvector;
        data = data*eigvector;
    end
    
    if strcmp(GENet.layer{stage}.type,'LDA')
        fprintf('the %d layer is LDA\n',stage);
%         if ~isfield(GENet.layer{stage},'Fisherface')
%             error('GENet.layer{%d}.ReducedDim is not found!',stage)
%         end
        options=[];
%         options.Fisherface=GENet.layer{stage}.Fisherface;
        [eigvector,eigvalue] = LDA(label,options,data);
        GENet.layer{stage}.eigvector=eigvector;
        data = data*eigvector;
    end
    
    if strcmp(GENet.layer{stage}.type,'KPCA')
        fprintf('the %d layer is KPCA\n',stage);
        if ~isfield(GENet.layer{stage},'ReducedDim')
            error('GENet.layer{%d}.ReducedDim is not found!',stage)
        end
        options=[];
        options.ReducedDim=GENet.layer{stage}.ReducedDim;
        [eigvector,eigvalue] = KPCA(data,options);
        temp=data;
        save temp temp
        data=constructKernel(data,data,options);
        GENet.layer{stage}.eigvector=eigvector;
        data = data*eigvector;
    end
    
    if strcmp(GENet.layer{stage}.type,'MFA')
        fprintf('the %d layer is MFA\n',stage);
        if ~isfield(GENet.layer{stage},'intraK')
            error('GENet.layer{%d}.intraK is not found!',stage)
        end
        if ~isfield(GENet.layer{stage},'interK')
            error('GENet.layer{%d}.interK is not found!',stage)
        end
        if ~isfield(GENet.layer{stage},'ReducedDim')
            error('GENet.layer{%d}.ReducedDim is not found!',stage)
        end
        if ~isfield(GENet.layer{stage},'Regu')
            error('GENet.layer{%d}.Regu is not found!',stage)
        end
        options = [];
        options.intraK = GENet.layer{stage}.intraK;  %越大越紧
        options.interK = GENet.layer{stage}.interK;  %越大越高
        options.ReducedDim = GENet.layer{stage}.ReducedDim;
        options.Regu = GENet.layer{stage}.Regu;
        [eigvector, eigvalue] = MFA(label, options, data);
        GENet.layer{stage}.eigvector=eigvector;
        data = data*eigvector;
    end
end