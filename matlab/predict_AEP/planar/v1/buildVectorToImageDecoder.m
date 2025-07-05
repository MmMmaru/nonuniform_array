function lgraph = buildVectorToImageDecoder(featureVectorSize, outputHeight, outputWidth, outputChannels)
% 构建一个从特征向量生成图像的解码器网络 (使用全连接层和转置卷积)。
%
% 参数:
%   featureVectorSize (标量): 输入特征向量的维度。
%   outputHeight (标量): 目标输出图像的高度。
%   outputWidth (标量): 目标输出图像的宽度。
%   outputChannels (标量): 目标输出图像的通道数 (例如 1 或 3)。
%
% 返回:
%   lgraph (layerGraph): 构建好的解码器网络。

    lgraph = layerGraph();
    
    % --- 设计决策 (可调整的超参数) ---
    initialDenseUnits = 256;       % 第一个全连接层的单元数 (示例)
    bottleneckSizeH = 4;           % 重塑后的初始高度 (必须是 2 的幂，且小于等于 outputHeight)
    bottleneckSizeW = 4;           % 重塑后的初始宽度 (必须是 2 的幂，且小于等于 outputWidth)
    bottleneckChannels = 512;      % 重塑后的初始通道数 (示例, 通常较大)
    numUpsampleLayers = log2(outputHeight / bottleneckSizeH); % 计算需要的上采样层数 (假设每次加倍)
    
    if outputHeight / bottleneckSizeH ~= 2^numUpsampleLayers || ...
       outputWidth / bottleneckSizeW ~= log2(outputWidth / bottleneckSizeW)^2 % 检查是否为2的幂次关系
        error('Output dimensions must be powers of 2 times the bottleneck dimensions.');
    end
    if mod(numUpsampleLayers, 1) ~= 0
        error('Calculated number of upsample layers is not an integer. Check dimensions.');
    end
    
    % 计算最后一个全连接层的输出大小，以便 reshape
    denseOutputUnits = bottleneckSizeH * bottleneckSizeW * bottleneckChannels;
    
    filterSize = 3; % 标准卷积核大小
    upsampleKernel = 4; % 转置卷积核大小 (通常用 2 或 4)
    upsampleStride = 2; % 转置卷积步长 (用于尺寸加倍)
    
    % --- 网络构建 ---
    
    % 1. 输入层
    layers = featureInputLayer(featureVectorSize, 'Name', 'inputFeature', 'Normalization', 'zscore'); % 或 'none'
    lgraph = addLayers(lgraph, layers);
    currentLayer = 'inputFeature';
    
    % 2. 全连接块
    fprintf('Building Fully Connected Block...\n');
    fcBlock = [
        fullyConnectedLayer(initialDenseUnits, 'Name', 'fc1')
        batchNormalizationLayer('Name','bn_fc1') % 可选，但常有帮助
        reluLayer('Name', 'relu_fc1')
        fullyConnectedLayer(denseOutputUnits, 'Name', 'fc_final') % 输出足够 reshape 的单元
        batchNormalizationLayer('Name','bn_fc_final')
        reluLayer('Name', 'relu_fc_final')
    ];
    lgraph = addLayers(lgraph, fcBlock);
    lgraph = connectLayers(lgraph, currentLayer, 'fc1');
    currentLayer = 'relu_fc_final';
    
    % 3. 重塑层 (Reshape)
    %    将 [denseOutputUnits] 转换为 [H, W, C]
    reshapeLayer_ = projectAndReshapeLayer([bottleneckSizeH, bottleneckSizeW, bottleneckChannels], Name='reshape');
    lgraph = addLayers(lgraph, reshapeLayer_);
    lgraph = connectLayers(lgraph, currentLayer, 'reshape');
    currentLayer = 'reshape';
    
    % 4. 上卷积块 (循环)
    fprintf('Building Up-Convolution Blocks...\n');
    numFilters = bottleneckChannels;
    for i = 1:numUpsampleLayers
        numFilters = numFilters / 2; % 每次上采样通道数减半 (常见做法)
        if numFilters < 16 % 避免通道数过少 (可调整)
            numFilters = 16;
        end
        fprintf(' UpConv Block %d, Output Filters: %d\n', i, numFilters);
    
        % 转置卷积 (上采样)
        upConvName = ['upConv' num2str(i)];
        upConvLayer = transposedConv2dLayer(upsampleKernel, numFilters, ...
            'Stride', upsampleStride, 'Cropping', 'same', 'Name', upConvName); % 'same' 裁剪尝试保持尺寸匹配
        lgraph = addLayers(lgraph, upConvLayer);
        lgraph = connectLayers(lgraph, currentLayer, upConvName);
        currentLayer = upConvName;
    
        % Batch Normalization + ReLU
        bnReluBlock1 = [
            batchNormalizationLayer('Name', ['bn_up' num2str(i)])
            reluLayer('Name', ['relu_up' num2str(i)])
        ];
        lgraph = addLayers(lgraph, bnReluBlock1);
        lgraph = connectLayers(lgraph, currentLayer, ['bn_up' num2str(i)]);
        currentLayer = ['relu_up' num2str(i)];
    
        % 标准卷积 (细化特征)
        convName = ['conv' num2str(i)];
        convLayer = convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', convName);
        lgraph = addLayers(lgraph, convLayer);
        lgraph = connectLayers(lgraph, currentLayer, convName);
        currentLayer = convName;
    
        % Batch Normalization + ReLU
        bnReluBlock2 = [
            batchNormalizationLayer('Name', ['bn_conv' num2str(i)])
            reluLayer('Name', ['relu_conv' num2str(i)])
        ];
        lgraph = addLayers(lgraph, bnReluBlock2);
        lgraph = connectLayers(lgraph, currentLayer, ['bn_conv' num2str(i)]);
        currentLayer = ['relu_conv' num2str(i)];
    end
    
    % 5. 最终输出卷积层
    fprintf('Building Final Output Layer...\n');
    finalConvLayer = convolution2dLayer(filterSize, outputChannels, 'Padding','same', 'Name', 'finalConv');
    lgraph = addLayers(lgraph, finalConvLayer);
    lgraph = connectLayers(lgraph, currentLayer, 'finalConv');
    currentLayer = 'finalConv';
    
    % 7. 输出层 (例如，用于计算损失的回归层)
    %    如果你在训练循环中自己计算损失，这层可以省略。
    %    如果使用 trainNetwork 并且是回归任务（预测像素值），则需要回归层。
    outputLayer = regressionLayer('Name', 'output');
    lgraph = addLayers(lgraph, outputLayer);
    lgraph = connectLayers(lgraph, currentLayer, 'output');
    
    fprintf('Decoder network built successfully.\n');

end
