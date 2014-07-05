function data=GENet_test(GENet,data)
for stage = 1:size(GENet.layer,2)
    if strcmp(GENet.layer{stage}.type,'KPCA')
        options=[];
        options.ReducedDim=GENet.layer{stage}.ReducedDim;
        load temp.mat
        Ktest = constructKernel(data,temp,options);
        data = Ktest*GENet.layer{stage}.eigvector;
    else
        data = data*GENet.layer{stage}.eigvector;
    end
end