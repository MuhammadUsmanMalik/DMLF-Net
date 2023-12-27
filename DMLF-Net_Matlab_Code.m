
params = load("D:\1_Papers Data\1_TEM Virus\4_Results\DMLFN\DMLF-Net\DMLF-Net_params.mat");
lgraph = layerGraph();

tempLayers = [
    imageInputLayer([224 224 3],"Name","input_1","Normalization","zscore","Mean",params.input_1.Mean,"StandardDeviation",params.input_1.StandardDeviation)
    convolution2dLayer([3 3],32,"Name","Conv1","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"WeightLearnRateFactor",0,"Bias",params.Conv1.Bias,"Weights",params.Conv1.Weights)
    batchNormalizationLayer("Name","bn_Conv1","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.bn_Conv1.Offset,"Scale",params.bn_Conv1.Scale,"TrainedMean",params.bn_Conv1.TrainedMean,"TrainedVariance",params.bn_Conv1.TrainedVariance)
    clippedReluLayer(6,"Name","Conv1_relu")
    groupedConvolution2dLayer([3 3],1,32,"Name","expanded_conv_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.expanded_conv_depthwise.Bias,"Weights",params.expanded_conv_depthwise.Weights)
    batchNormalizationLayer("Name","expanded_conv_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.expanded_conv_depthwise_BN.Offset,"Scale",params.expanded_conv_depthwise_BN.Scale,"TrainedMean",params.expanded_conv_depthwise_BN.TrainedMean,"TrainedVariance",params.expanded_conv_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","expanded_conv_depthwise_relu")
    convolution2dLayer([1 1],16,"Name","expanded_conv_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.expanded_conv_project.Bias,"Weights",params.expanded_conv_project.Weights)
    batchNormalizationLayer("Name","expanded_conv_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.expanded_conv_project_BN.Offset,"Scale",params.expanded_conv_project_BN.Scale,"TrainedMean",params.expanded_conv_project_BN.TrainedMean,"TrainedVariance",params.expanded_conv_project_BN.TrainedVariance)
    convolution2dLayer([1 1],96,"Name","block_1_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_1_expand.Bias,"Weights",params.block_1_expand.Weights)
    batchNormalizationLayer("Name","block_1_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_1_expand_BN.Offset,"Scale",params.block_1_expand_BN.Scale,"TrainedMean",params.block_1_expand_BN.TrainedMean,"TrainedVariance",params.block_1_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_1_expand_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],1,96,"Name","block_1_depthwise","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"WeightLearnRateFactor",0,"Bias",params.block_1_depthwise.Bias,"Weights",params.block_1_depthwise.Weights)
    batchNormalizationLayer("Name","block_1_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_1_depthwise_BN.Offset,"Scale",params.block_1_depthwise_BN.Scale,"TrainedMean",params.block_1_depthwise_BN.TrainedMean,"TrainedVariance",params.block_1_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_1_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_1_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_1_project.Bias,"Weights",params.block_1_project.Weights)
    batchNormalizationLayer("Name","block_1_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_1_project_BN.Offset,"Scale",params.block_1_project_BN.Scale,"TrainedMean",params.block_1_project_BN.TrainedMean,"TrainedVariance",params.block_1_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],144,"Name","block_2_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_2_expand.Bias,"Weights",params.block_2_expand.Weights)
    batchNormalizationLayer("Name","block_2_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_2_expand_BN.Offset,"Scale",params.block_2_expand_BN.Scale,"TrainedMean",params.block_2_expand_BN.TrainedMean,"TrainedVariance",params.block_2_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_2_expand_relu")
    groupedConvolution2dLayer([3 3],1,144,"Name","block_2_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_2_depthwise.Bias,"Weights",params.block_2_depthwise.Weights)
    batchNormalizationLayer("Name","block_2_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_2_depthwise_BN.Offset,"Scale",params.block_2_depthwise_BN.Scale,"TrainedMean",params.block_2_depthwise_BN.TrainedMean,"TrainedVariance",params.block_2_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_2_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_2_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_2_project.Bias,"Weights",params.block_2_project.Weights)
    batchNormalizationLayer("Name","block_2_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_2_project_BN.Offset,"Scale",params.block_2_project_BN.Scale,"TrainedMean",params.block_2_project_BN.TrainedMean,"TrainedVariance",params.block_2_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_2_add")
    convolution2dLayer([1 1],144,"Name","block_3_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_3_expand.Bias,"Weights",params.block_3_expand.Weights)
    batchNormalizationLayer("Name","block_3_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_3_expand_BN.Offset,"Scale",params.block_3_expand_BN.Scale,"TrainedMean",params.block_3_expand_BN.TrainedMean,"TrainedVariance",params.block_3_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_3_expand_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],1,144,"Name","block_3_depthwise","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"WeightLearnRateFactor",0,"Bias",params.block_3_depthwise.Bias,"Weights",params.block_3_depthwise.Weights)
    batchNormalizationLayer("Name","block_3_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_3_depthwise_BN.Offset,"Scale",params.block_3_depthwise_BN.Scale,"TrainedMean",params.block_3_depthwise_BN.TrainedMean,"TrainedVariance",params.block_3_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_3_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_3_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_3_project.Bias,"Weights",params.block_3_project.Weights)
    batchNormalizationLayer("Name","block_3_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_3_project_BN.Offset,"Scale",params.block_3_project_BN.Scale,"TrainedMean",params.block_3_project_BN.TrainedMean,"TrainedVariance",params.block_3_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_4_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_4_expand.Bias,"Weights",params.block_4_expand.Weights)
    batchNormalizationLayer("Name","block_4_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_4_expand_BN.Offset,"Scale",params.block_4_expand_BN.Scale,"TrainedMean",params.block_4_expand_BN.TrainedMean,"TrainedVariance",params.block_4_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_4_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_4_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_4_depthwise.Bias,"Weights",params.block_4_depthwise.Weights)
    batchNormalizationLayer("Name","block_4_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_4_depthwise_BN.Offset,"Scale",params.block_4_depthwise_BN.Scale,"TrainedMean",params.block_4_depthwise_BN.TrainedMean,"TrainedVariance",params.block_4_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_4_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_4_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_4_project.Bias,"Weights",params.block_4_project.Weights)
    batchNormalizationLayer("Name","block_4_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_4_project_BN.Offset,"Scale",params.block_4_project_BN.Scale,"TrainedMean",params.block_4_project_BN.TrainedMean,"TrainedVariance",params.block_4_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_4_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_5_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_5_expand.Bias,"Weights",params.block_5_expand.Weights)
    batchNormalizationLayer("Name","block_5_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_5_expand_BN.Offset,"Scale",params.block_5_expand_BN.Scale,"TrainedMean",params.block_5_expand_BN.TrainedMean,"TrainedVariance",params.block_5_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_5_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_5_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_5_depthwise.Bias,"Weights",params.block_5_depthwise.Weights)
    batchNormalizationLayer("Name","block_5_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_5_depthwise_BN.Offset,"Scale",params.block_5_depthwise_BN.Scale,"TrainedMean",params.block_5_depthwise_BN.TrainedMean,"TrainedVariance",params.block_5_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_5_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_5_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_5_project.Bias,"Weights",params.block_5_project.Weights)
    batchNormalizationLayer("Name","block_5_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_5_project_BN.Offset,"Scale",params.block_5_project_BN.Scale,"TrainedMean",params.block_5_project_BN.TrainedMean,"TrainedVariance",params.block_5_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_5_add")
    convolution2dLayer([1 1],192,"Name","block_6_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_6_expand.Bias,"Weights",params.block_6_expand.Weights)
    batchNormalizationLayer("Name","block_6_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_6_expand_BN.Offset,"Scale",params.block_6_expand_BN.Scale,"TrainedMean",params.block_6_expand_BN.TrainedMean,"TrainedVariance",params.block_6_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_6_expand_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],1,192,"Name","block_6_depthwise","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"WeightLearnRateFactor",0,"Bias",params.block_6_depthwise.Bias,"Weights",params.block_6_depthwise.Weights)
    batchNormalizationLayer("Name","block_6_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_6_depthwise_BN.Offset,"Scale",params.block_6_depthwise_BN.Scale,"TrainedMean",params.block_6_depthwise_BN.TrainedMean,"TrainedVariance",params.block_6_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_6_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_6_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_6_project.Bias,"Weights",params.block_6_project.Weights)
    batchNormalizationLayer("Name","block_6_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_6_project_BN.Offset,"Scale",params.block_6_project_BN.Scale,"TrainedMean",params.block_6_project_BN.TrainedMean,"TrainedVariance",params.block_6_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_7_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_7_expand.Bias,"Weights",params.block_7_expand.Weights)
    batchNormalizationLayer("Name","block_7_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_7_expand_BN.Offset,"Scale",params.block_7_expand_BN.Scale,"TrainedMean",params.block_7_expand_BN.TrainedMean,"TrainedVariance",params.block_7_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_7_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_7_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_7_depthwise.Bias,"Weights",params.block_7_depthwise.Weights)
    batchNormalizationLayer("Name","block_7_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_7_depthwise_BN.Offset,"Scale",params.block_7_depthwise_BN.Scale,"TrainedMean",params.block_7_depthwise_BN.TrainedMean,"TrainedVariance",params.block_7_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_7_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_7_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_7_project.Bias,"Weights",params.block_7_project.Weights)
    batchNormalizationLayer("Name","block_7_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_7_project_BN.Offset,"Scale",params.block_7_project_BN.Scale,"TrainedMean",params.block_7_project_BN.TrainedMean,"TrainedVariance",params.block_7_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_7_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_8_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_8_expand.Bias,"Weights",params.block_8_expand.Weights)
    batchNormalizationLayer("Name","block_8_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_8_expand_BN.Offset,"Scale",params.block_8_expand_BN.Scale,"TrainedMean",params.block_8_expand_BN.TrainedMean,"TrainedVariance",params.block_8_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_8_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_8_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_8_depthwise.Bias,"Weights",params.block_8_depthwise.Weights)
    batchNormalizationLayer("Name","block_8_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_8_depthwise_BN.Offset,"Scale",params.block_8_depthwise_BN.Scale,"TrainedMean",params.block_8_depthwise_BN.TrainedMean,"TrainedVariance",params.block_8_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_8_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_8_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_8_project.Bias,"Weights",params.block_8_project.Weights)
    batchNormalizationLayer("Name","block_8_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_8_project_BN.Offset,"Scale",params.block_8_project_BN.Scale,"TrainedMean",params.block_8_project_BN.TrainedMean,"TrainedVariance",params.block_8_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_8_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_9_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_9_expand.Bias,"Weights",params.block_9_expand.Weights)
    batchNormalizationLayer("Name","block_9_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_9_expand_BN.Offset,"Scale",params.block_9_expand_BN.Scale,"TrainedMean",params.block_9_expand_BN.TrainedMean,"TrainedVariance",params.block_9_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_9_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_9_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_9_depthwise.Bias,"Weights",params.block_9_depthwise.Weights)
    batchNormalizationLayer("Name","block_9_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_9_depthwise_BN.Offset,"Scale",params.block_9_depthwise_BN.Scale,"TrainedMean",params.block_9_depthwise_BN.TrainedMean,"TrainedVariance",params.block_9_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_9_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_9_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_9_project.Bias,"Weights",params.block_9_project.Weights)
    batchNormalizationLayer("Name","block_9_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_9_project_BN.Offset,"Scale",params.block_9_project_BN.Scale,"TrainedMean",params.block_9_project_BN.TrainedMean,"TrainedVariance",params.block_9_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_9_add")
    convolution2dLayer([1 1],384,"Name","block_10_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_10_expand.Bias,"Weights",params.block_10_expand.Weights)
    batchNormalizationLayer("Name","block_10_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_10_expand_BN.Offset,"Scale",params.block_10_expand_BN.Scale,"TrainedMean",params.block_10_expand_BN.TrainedMean,"TrainedVariance",params.block_10_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_10_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_10_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_10_depthwise.Bias,"Weights",params.block_10_depthwise.Weights)
    batchNormalizationLayer("Name","block_10_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_10_depthwise_BN.Offset,"Scale",params.block_10_depthwise_BN.Scale,"TrainedMean",params.block_10_depthwise_BN.TrainedMean,"TrainedVariance",params.block_10_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_10_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_10_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_10_project.Bias,"Weights",params.block_10_project.Weights)
    batchNormalizationLayer("Name","block_10_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_10_project_BN.Offset,"Scale",params.block_10_project_BN.Scale,"TrainedMean",params.block_10_project_BN.TrainedMean,"TrainedVariance",params.block_10_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_11_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_11_expand.Bias,"Weights",params.block_11_expand.Weights)
    batchNormalizationLayer("Name","block_11_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_11_expand_BN.Offset,"Scale",params.block_11_expand_BN.Scale,"TrainedMean",params.block_11_expand_BN.TrainedMean,"TrainedVariance",params.block_11_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_11_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_11_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_11_depthwise.Bias,"Weights",params.block_11_depthwise.Weights)
    batchNormalizationLayer("Name","block_11_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_11_depthwise_BN.Offset,"Scale",params.block_11_depthwise_BN.Scale,"TrainedMean",params.block_11_depthwise_BN.TrainedMean,"TrainedVariance",params.block_11_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_11_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_11_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_11_project.Bias,"Weights",params.block_11_project.Weights)
    batchNormalizationLayer("Name","block_11_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_11_project_BN.Offset,"Scale",params.block_11_project_BN.Scale,"TrainedMean",params.block_11_project_BN.TrainedMean,"TrainedVariance",params.block_11_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_11_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_12_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_12_expand.Bias,"Weights",params.block_12_expand.Weights)
    batchNormalizationLayer("Name","block_12_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_12_expand_BN.Offset,"Scale",params.block_12_expand_BN.Scale,"TrainedMean",params.block_12_expand_BN.TrainedMean,"TrainedVariance",params.block_12_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_12_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_12_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_12_depthwise.Bias,"Weights",params.block_12_depthwise.Weights)
    batchNormalizationLayer("Name","block_12_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_12_depthwise_BN.Offset,"Scale",params.block_12_depthwise_BN.Scale,"TrainedMean",params.block_12_depthwise_BN.TrainedMean,"TrainedVariance",params.block_12_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_12_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_12_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_12_project.Bias,"Weights",params.block_12_project.Weights)
    batchNormalizationLayer("Name","block_12_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_12_project_BN.Offset,"Scale",params.block_12_project_BN.Scale,"TrainedMean",params.block_12_project_BN.TrainedMean,"TrainedVariance",params.block_12_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_12_add")
    convolution2dLayer([1 1],576,"Name","block_13_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_13_expand.Bias,"Weights",params.block_13_expand.Weights)
    batchNormalizationLayer("Name","block_13_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_13_expand_BN.Offset,"Scale",params.block_13_expand_BN.Scale,"TrainedMean",params.block_13_expand_BN.TrainedMean,"TrainedVariance",params.block_13_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_13_expand_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],1,576,"Name","block_13_depthwise","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"WeightLearnRateFactor",0,"Bias",params.block_13_depthwise.Bias,"Weights",params.block_13_depthwise.Weights)
    batchNormalizationLayer("Name","block_13_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_13_depthwise_BN.Offset,"Scale",params.block_13_depthwise_BN.Scale,"TrainedMean",params.block_13_depthwise_BN.TrainedMean,"TrainedVariance",params.block_13_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_13_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_13_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_13_project.Bias,"Weights",params.block_13_project.Weights)
    batchNormalizationLayer("Name","block_13_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_13_project_BN.Offset,"Scale",params.block_13_project_BN.Scale,"TrainedMean",params.block_13_project_BN.TrainedMean,"TrainedVariance",params.block_13_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_14_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_14_expand.Bias,"Weights",params.block_14_expand.Weights)
    batchNormalizationLayer("Name","block_14_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_14_expand_BN.Offset,"Scale",params.block_14_expand_BN.Scale,"TrainedMean",params.block_14_expand_BN.TrainedMean,"TrainedVariance",params.block_14_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_14_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_14_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_14_depthwise.Bias,"Weights",params.block_14_depthwise.Weights)
    batchNormalizationLayer("Name","block_14_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_14_depthwise_BN.Offset,"Scale",params.block_14_depthwise_BN.Scale,"TrainedMean",params.block_14_depthwise_BN.TrainedMean,"TrainedVariance",params.block_14_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_14_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_14_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_14_project.Bias,"Weights",params.block_14_project.Weights)
    batchNormalizationLayer("Name","block_14_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_14_project_BN.Offset,"Scale",params.block_14_project_BN.Scale,"TrainedMean",params.block_14_project_BN.TrainedMean,"TrainedVariance",params.block_14_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_14_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_15_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_15_expand.Bias,"Weights",params.block_15_expand.Weights)
    batchNormalizationLayer("Name","block_15_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_15_expand_BN.Offset,"Scale",params.block_15_expand_BN.Scale,"TrainedMean",params.block_15_expand_BN.TrainedMean,"TrainedVariance",params.block_15_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_15_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_15_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_15_depthwise.Bias,"Weights",params.block_15_depthwise.Weights)
    batchNormalizationLayer("Name","block_15_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_15_depthwise_BN.Offset,"Scale",params.block_15_depthwise_BN.Scale,"TrainedMean",params.block_15_depthwise_BN.TrainedMean,"TrainedVariance",params.block_15_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_15_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_15_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_15_project.Bias,"Weights",params.block_15_project.Weights)
    batchNormalizationLayer("Name","block_15_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_15_project_BN.Offset,"Scale",params.block_15_project_BN.Scale,"TrainedMean",params.block_15_project_BN.TrainedMean,"TrainedVariance",params.block_15_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_15_add")
    convolution2dLayer([1 1],960,"Name","block_16_expand","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_16_expand.Bias,"Weights",params.block_16_expand.Weights)
    batchNormalizationLayer("Name","block_16_expand_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_16_expand_BN.Offset,"Scale",params.block_16_expand_BN.Scale,"TrainedMean",params.block_16_expand_BN.TrainedMean,"TrainedVariance",params.block_16_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_16_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_16_depthwise","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_16_depthwise.Bias,"Weights",params.block_16_depthwise.Weights)
    batchNormalizationLayer("Name","block_16_depthwise_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_16_depthwise_BN.Offset,"Scale",params.block_16_depthwise_BN.Scale,"TrainedMean",params.block_16_depthwise_BN.TrainedMean,"TrainedVariance",params.block_16_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_16_depthwise_relu")
    convolution2dLayer([1 1],320,"Name","block_16_project","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.block_16_project.Bias,"Weights",params.block_16_project.Weights)
    batchNormalizationLayer("Name","block_16_project_BN","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.block_16_project_BN.Offset,"Scale",params.block_16_project_BN.Scale,"TrainedMean",params.block_16_project_BN.TrainedMean,"TrainedVariance",params.block_16_project_BN.TrainedVariance)
    convolution2dLayer([1 1],1280,"Name","Conv_1","BiasLearnRateFactor",0,"WeightLearnRateFactor",0,"Bias",params.Conv_1.Bias,"Weights",params.Conv_1.Weights)
    batchNormalizationLayer("Name","Conv_1_bn","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.Conv_1_bn.Offset,"Scale",params.Conv_1_bn.Scale,"TrainedMean",params.Conv_1_bn.TrainedMean,"TrainedVariance",params.Conv_1_bn.TrainedVariance)
    clippedReluLayer(6,"Name","out_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalAveragePooling2dLayer("Name","global_average_pooling2d_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],8,"Name","conv3","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.conv3.Bias,"Weights",params.conv3.Weights)
    batchNormalizationLayer("Name","batchnorm3","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.batchnorm3.Offset,"Scale",params.batchnorm3.Scale,"TrainedMean",params.batchnorm3.TrainedMean,"TrainedVariance",params.batchnorm3.TrainedVariance)
    reluLayer("Name","relu3")
    fullyConnectedLayer(8,"Name","fc3","BiasLearnRateFactor",0,"WeightLearnRateFactor",0,"Bias",params.fc3.Bias,"Weights",params.fc3.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],4,"Name","conv2","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.conv2.Bias,"Weights",params.conv2.Weights)
    batchNormalizationLayer("Name","batchnorm2","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.batchnorm2.Offset,"Scale",params.batchnorm2.Scale,"TrainedMean",params.batchnorm2.TrainedMean,"TrainedVariance",params.batchnorm2.TrainedVariance)
    reluLayer("Name","relu2")
    fullyConnectedLayer(4,"Name","fc2","BiasLearnRateFactor",0,"WeightLearnRateFactor",0,"Bias",params.fc2.Bias,"Weights",params.fc2.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],2,"Name","conv1","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
    batchNormalizationLayer("Name","batchnorm1","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.batchnorm1.Offset,"Scale",params.batchnorm1.Scale,"TrainedMean",params.batchnorm1.TrainedMean,"TrainedVariance",params.batchnorm1.TrainedVariance)
    reluLayer("Name","relu1")
    fullyConnectedLayer(2,"Name","fc1","BiasLearnRateFactor",0,"WeightLearnRateFactor",0,"Bias",params.fc1.Bias,"Weights",params.fc1.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","conv4","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.conv4.Bias,"Weights",params.conv4.Weights)
    batchNormalizationLayer("Name","batchnorm4","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.batchnorm4.Offset,"Scale",params.batchnorm4.Scale,"TrainedMean",params.batchnorm4.TrainedMean,"TrainedVariance",params.batchnorm4.TrainedVariance)
    reluLayer("Name","relu4")
    fullyConnectedLayer(16,"Name","fc4","BiasLearnRateFactor",0,"WeightLearnRateFactor",0,"Bias",params.fc4.Bias,"Weights",params.fc4.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv5","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",0,"Bias",params.conv5.Bias,"Weights",params.conv5.Weights)
    batchNormalizationLayer("Name","batchnorm5","Epsilon",0.001,"OffsetLearnRateFactor",0,"ScaleLearnRateFactor",0,"Offset",params.batchnorm5.Offset,"Scale",params.batchnorm5.Scale,"TrainedMean",params.batchnorm5.TrainedMean,"TrainedVariance",params.batchnorm5.TrainedVariance)
    reluLayer("Name","relu5")
    fullyConnectedLayer(32,"Name","fc5","BiasLearnRateFactor",0,"WeightLearnRateFactor",0,"Bias",params.fc5.Bias,"Weights",params.fc5.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(5,"Name","depthcat1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","conv6","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10,"Bias",params.conv6.Bias,"Weights",params.conv6.Weights)
    batchNormalizationLayer("Name","batchnorm6","Offset",params.batchnorm6.Offset,"Scale",params.batchnorm6.Scale,"TrainedMean",params.batchnorm6.TrainedMean,"TrainedVariance",params.batchnorm6.TrainedVariance)
    reluLayer("Name","relu6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","conv7","BiasLearnRateFactor",0,"DilationFactor",[6 6],"Padding","same","WeightLearnRateFactor",10,"Bias",params.conv7.Bias,"Weights",params.conv7.Weights)
    batchNormalizationLayer("Name","batchnorm7","Offset",params.batchnorm7.Offset,"Scale",params.batchnorm7.Scale,"TrainedMean",params.batchnorm7.TrainedMean,"TrainedVariance",params.batchnorm7.TrainedVariance)
    reluLayer("Name","relu7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","conv8","BiasLearnRateFactor",0,"DilationFactor",[12 12],"Padding","same","WeightLearnRateFactor",10,"Bias",params.conv8.Bias,"Weights",params.conv8.Weights)
    batchNormalizationLayer("Name","batchnorm8","Offset",params.batchnorm8.Offset,"Scale",params.batchnorm8.Scale,"TrainedMean",params.batchnorm8.TrainedMean,"TrainedVariance",params.batchnorm8.TrainedVariance)
    reluLayer("Name","relu8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","conv9","BiasLearnRateFactor",0,"DilationFactor",[18 18],"Padding","same","WeightLearnRateFactor",10,"Bias",params.conv9.Bias,"Weights",params.conv9.Weights)
    batchNormalizationLayer("Name","batchnorm9","Offset",params.batchnorm9.Offset,"Scale",params.batchnorm9.Scale,"TrainedMean",params.batchnorm9.TrainedMean,"TrainedVariance",params.batchnorm9.TrainedVariance)
    reluLayer("Name","relu9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","conv10","BiasLearnRateFactor",0,"DilationFactor",[24 24],"Padding","same","WeightLearnRateFactor",10,"Bias",params.conv10.Bias,"Weights",params.conv10.Weights)
    batchNormalizationLayer("Name","batchnorm10","Offset",params.batchnorm10.Offset,"Scale",params.batchnorm10.Scale,"TrainedMean",params.batchnorm10.TrainedMean,"TrainedVariance",params.batchnorm10.TrainedVariance)
    reluLayer("Name","relu13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(5,"Name","depthcat3")
    fullyConnectedLayer(1280,"Name","fc6","Bias",params.fc6.Bias,"Weights",params.fc6.Weights)
    reluLayer("Name","relu10")
    fullyConnectedLayer(1280,"Name","fc7","Bias",params.fc7.Bias,"Weights",params.fc7.Weights)
    reluLayer("Name","relu11")
    fullyConnectedLayer(22,"Name","fc8","Bias",params.fc8.Bias,"Weights",params.fc8.Weights)
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput","Classes",params.classoutput.Classes)];
lgraph = addLayers(lgraph,tempLayers);
clear tempLayers;

lgraph = connectLayers(lgraph,"block_1_expand_relu","block_1_depthwise");
lgraph = connectLayers(lgraph,"block_1_expand_relu","conv1");
lgraph = connectLayers(lgraph,"block_1_project_BN","block_2_expand");
lgraph = connectLayers(lgraph,"block_1_project_BN","block_2_add/in2");
lgraph = connectLayers(lgraph,"block_2_project_BN","block_2_add/in1");
lgraph = connectLayers(lgraph,"block_3_expand_relu","block_3_depthwise");
lgraph = connectLayers(lgraph,"block_3_expand_relu","conv2");
lgraph = connectLayers(lgraph,"block_3_project_BN","block_4_expand");
lgraph = connectLayers(lgraph,"block_3_project_BN","block_4_add/in2");
lgraph = connectLayers(lgraph,"block_4_project_BN","block_4_add/in1");
lgraph = connectLayers(lgraph,"block_4_add","block_5_expand");
lgraph = connectLayers(lgraph,"block_4_add","block_5_add/in2");
lgraph = connectLayers(lgraph,"block_5_project_BN","block_5_add/in1");
lgraph = connectLayers(lgraph,"block_6_expand_relu","block_6_depthwise");
lgraph = connectLayers(lgraph,"block_6_expand_relu","conv3");
lgraph = connectLayers(lgraph,"block_6_project_BN","block_7_expand");
lgraph = connectLayers(lgraph,"block_6_project_BN","block_7_add/in2");
lgraph = connectLayers(lgraph,"block_7_project_BN","block_7_add/in1");
lgraph = connectLayers(lgraph,"block_7_add","block_8_expand");
lgraph = connectLayers(lgraph,"block_7_add","block_8_add/in2");
lgraph = connectLayers(lgraph,"block_8_project_BN","block_8_add/in1");
lgraph = connectLayers(lgraph,"block_8_add","block_9_expand");
lgraph = connectLayers(lgraph,"block_8_add","block_9_add/in2");
lgraph = connectLayers(lgraph,"block_9_project_BN","block_9_add/in1");
lgraph = connectLayers(lgraph,"block_10_project_BN","block_11_expand");
lgraph = connectLayers(lgraph,"block_10_project_BN","block_11_add/in2");
lgraph = connectLayers(lgraph,"block_11_project_BN","block_11_add/in1");
lgraph = connectLayers(lgraph,"block_11_add","block_12_expand");
lgraph = connectLayers(lgraph,"block_11_add","block_12_add/in2");
lgraph = connectLayers(lgraph,"block_12_project_BN","block_12_add/in1");
lgraph = connectLayers(lgraph,"block_13_expand_relu","block_13_depthwise");
lgraph = connectLayers(lgraph,"block_13_expand_relu","conv4");
lgraph = connectLayers(lgraph,"block_13_project_BN","block_14_expand");
lgraph = connectLayers(lgraph,"block_13_project_BN","block_14_add/in2");
lgraph = connectLayers(lgraph,"block_14_project_BN","block_14_add/in1");
lgraph = connectLayers(lgraph,"block_14_add","block_15_expand");
lgraph = connectLayers(lgraph,"block_14_add","block_15_add/in2");
lgraph = connectLayers(lgraph,"block_15_project_BN","block_15_add/in1");
lgraph = connectLayers(lgraph,"out_relu","global_average_pooling2d_1");
lgraph = connectLayers(lgraph,"out_relu","conv5");
lgraph = connectLayers(lgraph,"global_average_pooling2d_1","depthcat2/in1");
lgraph = connectLayers(lgraph,"fc3","depthcat1/in5");
lgraph = connectLayers(lgraph,"fc2","depthcat1/in4");
lgraph = connectLayers(lgraph,"fc1","depthcat1/in3");
lgraph = connectLayers(lgraph,"fc4","depthcat1/in1");
lgraph = connectLayers(lgraph,"fc5","depthcat1/in2");
lgraph = connectLayers(lgraph,"depthcat1","depthcat2/in2");
lgraph = connectLayers(lgraph,"depthcat2","conv6");
lgraph = connectLayers(lgraph,"depthcat2","conv7");
lgraph = connectLayers(lgraph,"depthcat2","conv8");
lgraph = connectLayers(lgraph,"depthcat2","conv9");
lgraph = connectLayers(lgraph,"depthcat2","conv10");
lgraph = connectLayers(lgraph,"relu6","depthcat3/in1");
lgraph = connectLayers(lgraph,"relu7","depthcat3/in2");
lgraph = connectLayers(lgraph,"relu8","depthcat3/in3");
lgraph = connectLayers(lgraph,"relu9","depthcat3/in4");
lgraph = connectLayers(lgraph,"relu13","depthcat3/in5");
