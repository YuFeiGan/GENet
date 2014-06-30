function data=GENet_test(GENet,data)
for stage = 1:size(GENet.layer,1)
    data = data*GENet.layer{stage}.eigvector;
end