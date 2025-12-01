混合精度 QAT 及 config 配置方法：
https://github.com/AXERA-TECH/QAT.axera/tree/feat/support_mixed_precision


---
以 resnet50 为例，假设要让前两个 block 配置为 U16，config.json 配置如下: 
```
{
    "global_config": {
        "is_symmetric": false,
        "input": {
            "dtype": "U8",
            "qmin": 0,
            "qmax": 255
        },
        "weight": {
            "dtype": "S8",
            "qmin": -127,
            "qmax": 127
        }
    },
    "regional_configs": [
        {
            "module_names": ["conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5"],
            "module_type": "conv",
            "module_config": {
                "is_symmetric": false,
                "input": {
                    "dtype": "U16",
                    "qmin": 0,
                    "qmax": 65535
                },
                "weight": {
                    "dtype": "S8",
                    "qmin": -127,
                    "qmax": 127
                }
            }
        },
        {
            "module_names": ["add__5", "add__9"],
            "module_type": "add",
            "module_config": {
                "is_symmetric": false,
                "input": {
                    "dtype": "U16",
                    "qmin": 0,
                    "qmax": 65535
                }
            }
        }
    ]
}


```

其中 module_names 需要到 fx graph 里找，具体方法如下
浮点模型在 export 后得到 exported_model
`exported_model = torch.export.export_for_training(float_model, example_inputs).module()`
打印它的 graph
`print(exported_model.graph)`
可以得到如下 node 信息
```
graph():
    %conv1_weight : [num_users=1] = get_attr[target=conv1.weight]
    %bn1_weight : [num_users=1] = get_attr[target=bn1.weight]
    %bn1_bias : [num_users=1] = get_attr[target=bn1.bias]
    %layer1_0_conv1_weight : [num_users=1] = get_attr[target=layer1.0.conv1.weight]
    %layer1_0_bn1_weight : [num_users=1] = get_attr[target=layer1.0.bn1.weight]
    %layer1_0_bn1_bias : [num_users=1] = get_attr[target=layer1.0.bn1.bias]
    %layer1_0_conv2_weight : [num_users=1] = get_attr[target=layer1.0.conv2.weight]
    %layer1_0_bn2_weight : [num_users=1] = get_attr[target=layer1.0.bn2.weight]
    %layer1_0_bn2_bias : [num_users=1] = get_attr[target=layer1.0.bn2.bias]
    %layer1_0_conv3_weight : [num_users=1] = get_attr[target=layer1.0.conv3.weight]
    %layer1_0_bn3_weight : [num_users=1] = get_attr[target=layer1.0.bn3.weight]
    %layer1_0_bn3_bias : [num_users=1] = get_attr[target=layer1.0.bn3.bias]
    %layer1_0_downsample_0_weight : [num_users=1] = get_attr[target=layer1.0.downsample.0.weight]
    %layer1_0_downsample_1_weight : [num_users=1] = get_attr[target=layer1.0.downsample.1.weight]
    %layer1_0_downsample_1_bias : [num_users=1] = get_attr[target=layer1.0.downsample.1.bias]
    %layer1_1_conv1_weight : [num_users=1] = get_attr[target=layer1.1.conv1.weight]
    %layer1_1_bn1_weight : [num_users=1] = get_attr[target=layer1.1.bn1.weight]
    %layer1_1_bn1_bias : [num_users=1] = get_attr[target=layer1.1.bn1.bias]
    %layer1_1_conv2_weight : [num_users=1] = get_attr[target=layer1.1.conv2.weight]
    %layer1_1_bn2_weight : [num_users=1] = get_attr[target=layer1.1.bn2.weight]
    %layer1_1_bn2_bias : [num_users=1] = get_attr[target=layer1.1.bn2.bias]
    %layer1_1_conv3_weight : [num_users=1] = get_attr[target=layer1.1.conv3.weight]
    %layer1_1_bn3_weight : [num_users=1] = get_attr[target=layer1.1.bn3.weight]
    %layer1_1_bn3_bias : [num_users=1] = get_attr[target=layer1.1.bn3.bias]
    %layer1_2_conv1_weight : [num_users=1] = get_attr[target=layer1.2.conv1.weight]
    %layer1_2_bn1_weight : [num_users=1] = get_attr[target=layer1.2.bn1.weight]
    %layer1_2_bn1_bias : [num_users=1] = get_attr[target=layer1.2.bn1.bias]
    %layer1_2_conv2_weight : [num_users=1] = get_attr[target=layer1.2.conv2.weight]
    %layer1_2_bn2_weight : [num_users=1] = get_attr[target=layer1.2.bn2.weight]
    %layer1_2_bn2_bias : [num_users=1] = get_attr[target=layer1.2.bn2.bias]
    %layer1_2_conv3_weight : [num_users=1] = get_attr[target=layer1.2.conv3.weight]
    %layer1_2_bn3_weight : [num_users=1] = get_attr[target=layer1.2.bn3.weight]
    %layer1_2_bn3_bias : [num_users=1] = get_attr[target=layer1.2.bn3.bias]
    %layer2_0_conv1_weight : [num_users=1] = get_attr[target=layer2.0.conv1.weight]
    %layer2_0_bn1_weight : [num_users=1] = get_attr[target=layer2.0.bn1.weight]
    %layer2_0_bn1_bias : [num_users=1] = get_attr[target=layer2.0.bn1.bias]
    %layer2_0_conv2_weight : [num_users=1] = get_attr[target=layer2.0.conv2.weight]
    %layer2_0_bn2_weight : [num_users=1] = get_attr[target=layer2.0.bn2.weight]
    %layer2_0_bn2_bias : [num_users=1] = get_attr[target=layer2.0.bn2.bias]
    %layer2_0_conv3_weight : [num_users=1] = get_attr[target=layer2.0.conv3.weight]
    %layer2_0_bn3_weight : [num_users=1] = get_attr[target=layer2.0.bn3.weight]
    %layer2_0_bn3_bias : [num_users=1] = get_attr[target=layer2.0.bn3.bias]
    %layer2_0_downsample_0_weight : [num_users=1] = get_attr[target=layer2.0.downsample.0.weight]
    %layer2_0_downsample_1_weight : [num_users=1] = get_attr[target=layer2.0.downsample.1.weight]
    %layer2_0_downsample_1_bias : [num_users=1] = get_attr[target=layer2.0.downsample.1.bias]
    %layer2_1_conv1_weight : [num_users=1] = get_attr[target=layer2.1.conv1.weight]
    %layer2_1_bn1_weight : [num_users=1] = get_attr[target=layer2.1.bn1.weight]
    %layer2_1_bn1_bias : [num_users=1] = get_attr[target=layer2.1.bn1.bias]
    %layer2_1_conv2_weight : [num_users=1] = get_attr[target=layer2.1.conv2.weight]
    %layer2_1_bn2_weight : [num_users=1] = get_attr[target=layer2.1.bn2.weight]
    %layer2_1_bn2_bias : [num_users=1] = get_attr[target=layer2.1.bn2.bias]
    %layer2_1_conv3_weight : [num_users=1] = get_attr[target=layer2.1.conv3.weight]
    %layer2_1_bn3_weight : [num_users=1] = get_attr[target=layer2.1.bn3.weight]
    %layer2_1_bn3_bias : [num_users=1] = get_attr[target=layer2.1.bn3.bias]
    %layer2_2_conv1_weight : [num_users=1] = get_attr[target=layer2.2.conv1.weight]
    %layer2_2_bn1_weight : [num_users=1] = get_attr[target=layer2.2.bn1.weight]
    %layer2_2_bn1_bias : [num_users=1] = get_attr[target=layer2.2.bn1.bias]
    %layer2_2_conv2_weight : [num_users=1] = get_attr[target=layer2.2.conv2.weight]
    %layer2_2_bn2_weight : [num_users=1] = get_attr[target=layer2.2.bn2.weight]
    %layer2_2_bn2_bias : [num_users=1] = get_attr[target=layer2.2.bn2.bias]
    %layer2_2_conv3_weight : [num_users=1] = get_attr[target=layer2.2.conv3.weight]
    %layer2_2_bn3_weight : [num_users=1] = get_attr[target=layer2.2.bn3.weight]
    %layer2_2_bn3_bias : [num_users=1] = get_attr[target=layer2.2.bn3.bias]
    %layer2_3_conv1_weight : [num_users=1] = get_attr[target=layer2.3.conv1.weight]
    %layer2_3_bn1_weight : [num_users=1] = get_attr[target=layer2.3.bn1.weight]
    %layer2_3_bn1_bias : [num_users=1] = get_attr[target=layer2.3.bn1.bias]
    %layer2_3_conv2_weight : [num_users=1] = get_attr[target=layer2.3.conv2.weight]
    %layer2_3_bn2_weight : [num_users=1] = get_attr[target=layer2.3.bn2.weight]
    %layer2_3_bn2_bias : [num_users=1] = get_attr[target=layer2.3.bn2.bias]
    %layer2_3_conv3_weight : [num_users=1] = get_attr[target=layer2.3.conv3.weight]
    %layer2_3_bn3_weight : [num_users=1] = get_attr[target=layer2.3.bn3.weight]
    %layer2_3_bn3_bias : [num_users=1] = get_attr[target=layer2.3.bn3.bias]
    %layer3_0_conv1_weight : [num_users=1] = get_attr[target=layer3.0.conv1.weight]
    %layer3_0_bn1_weight : [num_users=1] = get_attr[target=layer3.0.bn1.weight]
    %layer3_0_bn1_bias : [num_users=1] = get_attr[target=layer3.0.bn1.bias]
    %layer3_0_conv2_weight : [num_users=1] = get_attr[target=layer3.0.conv2.weight]
    %layer3_0_bn2_weight : [num_users=1] = get_attr[target=layer3.0.bn2.weight]
    %layer3_0_bn2_bias : [num_users=1] = get_attr[target=layer3.0.bn2.bias]
    %layer3_0_conv3_weight : [num_users=1] = get_attr[target=layer3.0.conv3.weight]
    %layer3_0_bn3_weight : [num_users=1] = get_attr[target=layer3.0.bn3.weight]
    %layer3_0_bn3_bias : [num_users=1] = get_attr[target=layer3.0.bn3.bias]
    %layer3_0_downsample_0_weight : [num_users=1] = get_attr[target=layer3.0.downsample.0.weight]
    %layer3_0_downsample_1_weight : [num_users=1] = get_attr[target=layer3.0.downsample.1.weight]
    %layer3_0_downsample_1_bias : [num_users=1] = get_attr[target=layer3.0.downsample.1.bias]
    %layer3_1_conv1_weight : [num_users=1] = get_attr[target=layer3.1.conv1.weight]
    %layer3_1_bn1_weight : [num_users=1] = get_attr[target=layer3.1.bn1.weight]
    %layer3_1_bn1_bias : [num_users=1] = get_attr[target=layer3.1.bn1.bias]
    %layer3_1_conv2_weight : [num_users=1] = get_attr[target=layer3.1.conv2.weight]
    %layer3_1_bn2_weight : [num_users=1] = get_attr[target=layer3.1.bn2.weight]
    %layer3_1_bn2_bias : [num_users=1] = get_attr[target=layer3.1.bn2.bias]
    %layer3_1_conv3_weight : [num_users=1] = get_attr[target=layer3.1.conv3.weight]
    %layer3_1_bn3_weight : [num_users=1] = get_attr[target=layer3.1.bn3.weight]
    %layer3_1_bn3_bias : [num_users=1] = get_attr[target=layer3.1.bn3.bias]
    %layer3_2_conv1_weight : [num_users=1] = get_attr[target=layer3.2.conv1.weight]
    %layer3_2_bn1_weight : [num_users=1] = get_attr[target=layer3.2.bn1.weight]
    %layer3_2_bn1_bias : [num_users=1] = get_attr[target=layer3.2.bn1.bias]
    %layer3_2_conv2_weight : [num_users=1] = get_attr[target=layer3.2.conv2.weight]
    %layer3_2_bn2_weight : [num_users=1] = get_attr[target=layer3.2.bn2.weight]
    %layer3_2_bn2_bias : [num_users=1] = get_attr[target=layer3.2.bn2.bias]
    %layer3_2_conv3_weight : [num_users=1] = get_attr[target=layer3.2.conv3.weight]
    %layer3_2_bn3_weight : [num_users=1] = get_attr[target=layer3.2.bn3.weight]
    %layer3_2_bn3_bias : [num_users=1] = get_attr[target=layer3.2.bn3.bias]
    %layer3_3_conv1_weight : [num_users=1] = get_attr[target=layer3.3.conv1.weight]
    %layer3_3_bn1_weight : [num_users=1] = get_attr[target=layer3.3.bn1.weight]
    %layer3_3_bn1_bias : [num_users=1] = get_attr[target=layer3.3.bn1.bias]
    %layer3_3_conv2_weight : [num_users=1] = get_attr[target=layer3.3.conv2.weight]
    %layer3_3_bn2_weight : [num_users=1] = get_attr[target=layer3.3.bn2.weight]
    %layer3_3_bn2_bias : [num_users=1] = get_attr[target=layer3.3.bn2.bias]
    %layer3_3_conv3_weight : [num_users=1] = get_attr[target=layer3.3.conv3.weight]
    %layer3_3_bn3_weight : [num_users=1] = get_attr[target=layer3.3.bn3.weight]
    %layer3_3_bn3_bias : [num_users=1] = get_attr[target=layer3.3.bn3.bias]
    %layer3_4_conv1_weight : [num_users=1] = get_attr[target=layer3.4.conv1.weight]
    %layer3_4_bn1_weight : [num_users=1] = get_attr[target=layer3.4.bn1.weight]
    %layer3_4_bn1_bias : [num_users=1] = get_attr[target=layer3.4.bn1.bias]
    %layer3_4_conv2_weight : [num_users=1] = get_attr[target=layer3.4.conv2.weight]
    %layer3_4_bn2_weight : [num_users=1] = get_attr[target=layer3.4.bn2.weight]
    %layer3_4_bn2_bias : [num_users=1] = get_attr[target=layer3.4.bn2.bias]
    %layer3_4_conv3_weight : [num_users=1] = get_attr[target=layer3.4.conv3.weight]
    %layer3_4_bn3_weight : [num_users=1] = get_attr[target=layer3.4.bn3.weight]
    %layer3_4_bn3_bias : [num_users=1] = get_attr[target=layer3.4.bn3.bias]
    %layer3_5_conv1_weight : [num_users=1] = get_attr[target=layer3.5.conv1.weight]
    %layer3_5_bn1_weight : [num_users=1] = get_attr[target=layer3.5.bn1.weight]
    %layer3_5_bn1_bias : [num_users=1] = get_attr[target=layer3.5.bn1.bias]
    %layer3_5_conv2_weight : [num_users=1] = get_attr[target=layer3.5.conv2.weight]
    %layer3_5_bn2_weight : [num_users=1] = get_attr[target=layer3.5.bn2.weight]
    %layer3_5_bn2_bias : [num_users=1] = get_attr[target=layer3.5.bn2.bias]
    %layer3_5_conv3_weight : [num_users=1] = get_attr[target=layer3.5.conv3.weight]
    %layer3_5_bn3_weight : [num_users=1] = get_attr[target=layer3.5.bn3.weight]
    %layer3_5_bn3_bias : [num_users=1] = get_attr[target=layer3.5.bn3.bias]
    %layer4_0_conv1_weight : [num_users=1] = get_attr[target=layer4.0.conv1.weight]
    %layer4_0_bn1_weight : [num_users=1] = get_attr[target=layer4.0.bn1.weight]
    %layer4_0_bn1_bias : [num_users=1] = get_attr[target=layer4.0.bn1.bias]
    %layer4_0_conv2_weight : [num_users=1] = get_attr[target=layer4.0.conv2.weight]
    %layer4_0_bn2_weight : [num_users=1] = get_attr[target=layer4.0.bn2.weight]
    %layer4_0_bn2_bias : [num_users=1] = get_attr[target=layer4.0.bn2.bias]
    %layer4_0_conv3_weight : [num_users=1] = get_attr[target=layer4.0.conv3.weight]
    %layer4_0_bn3_weight : [num_users=1] = get_attr[target=layer4.0.bn3.weight]
    %layer4_0_bn3_bias : [num_users=1] = get_attr[target=layer4.0.bn3.bias]
    %layer4_0_downsample_0_weight : [num_users=1] = get_attr[target=layer4.0.downsample.0.weight]
    %layer4_0_downsample_1_weight : [num_users=1] = get_attr[target=layer4.0.downsample.1.weight]
    %layer4_0_downsample_1_bias : [num_users=1] = get_attr[target=layer4.0.downsample.1.bias]
    %layer4_1_conv1_weight : [num_users=1] = get_attr[target=layer4.1.conv1.weight]
    %layer4_1_bn1_weight : [num_users=1] = get_attr[target=layer4.1.bn1.weight]
    %layer4_1_bn1_bias : [num_users=1] = get_attr[target=layer4.1.bn1.bias]
    %layer4_1_conv2_weight : [num_users=1] = get_attr[target=layer4.1.conv2.weight]
    %layer4_1_bn2_weight : [num_users=1] = get_attr[target=layer4.1.bn2.weight]
    %layer4_1_bn2_bias : [num_users=1] = get_attr[target=layer4.1.bn2.bias]
    %layer4_1_conv3_weight : [num_users=1] = get_attr[target=layer4.1.conv3.weight]
    %layer4_1_bn3_weight : [num_users=1] = get_attr[target=layer4.1.bn3.weight]
    %layer4_1_bn3_bias : [num_users=1] = get_attr[target=layer4.1.bn3.bias]
    %layer4_2_conv1_weight : [num_users=1] = get_attr[target=layer4.2.conv1.weight]
    %layer4_2_bn1_weight : [num_users=1] = get_attr[target=layer4.2.bn1.weight]
    %layer4_2_bn1_bias : [num_users=1] = get_attr[target=layer4.2.bn1.bias]
    %layer4_2_conv2_weight : [num_users=1] = get_attr[target=layer4.2.conv2.weight]
    %layer4_2_bn2_weight : [num_users=1] = get_attr[target=layer4.2.bn2.weight]
    %layer4_2_bn2_bias : [num_users=1] = get_attr[target=layer4.2.bn2.bias]
    %layer4_2_conv3_weight : [num_users=1] = get_attr[target=layer4.2.conv3.weight]
    %layer4_2_bn3_weight : [num_users=1] = get_attr[target=layer4.2.bn3.weight]
    %layer4_2_bn3_bias : [num_users=1] = get_attr[target=layer4.2.bn3.bias]
    %fc_weight : [num_users=1] = get_attr[target=fc.weight]
    %fc_bias : [num_users=1] = get_attr[target=fc.bias]
    %bn1_running_mean : [num_users=1] = get_attr[target=bn1.running_mean]
    %bn1_running_var : [num_users=1] = get_attr[target=bn1.running_var]
    %bn1_num_batches_tracked : [num_users=1] = get_attr[target=bn1.num_batches_tracked]
    %layer1_0_bn1_running_mean : [num_users=1] = get_attr[target=layer1.0.bn1.running_mean]
    %layer1_0_bn1_running_var : [num_users=1] = get_attr[target=layer1.0.bn1.running_var]
    %layer1_0_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer1.0.bn1.num_batches_tracked]
    %layer1_0_bn2_running_mean : [num_users=1] = get_attr[target=layer1.0.bn2.running_mean]
    %layer1_0_bn2_running_var : [num_users=1] = get_attr[target=layer1.0.bn2.running_var]
    %layer1_0_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer1.0.bn2.num_batches_tracked]
    %layer1_0_bn3_running_mean : [num_users=1] = get_attr[target=layer1.0.bn3.running_mean]
    %layer1_0_bn3_running_var : [num_users=1] = get_attr[target=layer1.0.bn3.running_var]
    %layer1_0_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer1.0.bn3.num_batches_tracked]
    %layer1_0_downsample_1_running_mean : [num_users=1] = get_attr[target=layer1.0.downsample.1.running_mean]
    %layer1_0_downsample_1_running_var : [num_users=1] = get_attr[target=layer1.0.downsample.1.running_var]
    %layer1_0_downsample_1_num_batches_tracked : [num_users=1] = get_attr[target=layer1.0.downsample.1.num_batches_tracked]
    %layer1_1_bn1_running_mean : [num_users=1] = get_attr[target=layer1.1.bn1.running_mean]
    %layer1_1_bn1_running_var : [num_users=1] = get_attr[target=layer1.1.bn1.running_var]
    %layer1_1_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer1.1.bn1.num_batches_tracked]
    %layer1_1_bn2_running_mean : [num_users=1] = get_attr[target=layer1.1.bn2.running_mean]
    %layer1_1_bn2_running_var : [num_users=1] = get_attr[target=layer1.1.bn2.running_var]
    %layer1_1_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer1.1.bn2.num_batches_tracked]
    %layer1_1_bn3_running_mean : [num_users=1] = get_attr[target=layer1.1.bn3.running_mean]
    %layer1_1_bn3_running_var : [num_users=1] = get_attr[target=layer1.1.bn3.running_var]
    %layer1_1_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer1.1.bn3.num_batches_tracked]
    %layer1_2_bn1_running_mean : [num_users=1] = get_attr[target=layer1.2.bn1.running_mean]
    %layer1_2_bn1_running_var : [num_users=1] = get_attr[target=layer1.2.bn1.running_var]
    %layer1_2_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer1.2.bn1.num_batches_tracked]
    %layer1_2_bn2_running_mean : [num_users=1] = get_attr[target=layer1.2.bn2.running_mean]
    %layer1_2_bn2_running_var : [num_users=1] = get_attr[target=layer1.2.bn2.running_var]
    %layer1_2_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer1.2.bn2.num_batches_tracked]
    %layer1_2_bn3_running_mean : [num_users=1] = get_attr[target=layer1.2.bn3.running_mean]
    %layer1_2_bn3_running_var : [num_users=1] = get_attr[target=layer1.2.bn3.running_var]
    %layer1_2_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer1.2.bn3.num_batches_tracked]
    %layer2_0_bn1_running_mean : [num_users=1] = get_attr[target=layer2.0.bn1.running_mean]
    %layer2_0_bn1_running_var : [num_users=1] = get_attr[target=layer2.0.bn1.running_var]
    %layer2_0_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer2.0.bn1.num_batches_tracked]
    %layer2_0_bn2_running_mean : [num_users=1] = get_attr[target=layer2.0.bn2.running_mean]
    %layer2_0_bn2_running_var : [num_users=1] = get_attr[target=layer2.0.bn2.running_var]
    %layer2_0_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer2.0.bn2.num_batches_tracked]
    %layer2_0_bn3_running_mean : [num_users=1] = get_attr[target=layer2.0.bn3.running_mean]
    %layer2_0_bn3_running_var : [num_users=1] = get_attr[target=layer2.0.bn3.running_var]
    %layer2_0_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer2.0.bn3.num_batches_tracked]
    %layer2_0_downsample_1_running_mean : [num_users=1] = get_attr[target=layer2.0.downsample.1.running_mean]
    %layer2_0_downsample_1_running_var : [num_users=1] = get_attr[target=layer2.0.downsample.1.running_var]
    %layer2_0_downsample_1_num_batches_tracked : [num_users=1] = get_attr[target=layer2.0.downsample.1.num_batches_tracked]
    %layer2_1_bn1_running_mean : [num_users=1] = get_attr[target=layer2.1.bn1.running_mean]
    %layer2_1_bn1_running_var : [num_users=1] = get_attr[target=layer2.1.bn1.running_var]
    %layer2_1_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer2.1.bn1.num_batches_tracked]
    %layer2_1_bn2_running_mean : [num_users=1] = get_attr[target=layer2.1.bn2.running_mean]
    %layer2_1_bn2_running_var : [num_users=1] = get_attr[target=layer2.1.bn2.running_var]
    %layer2_1_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer2.1.bn2.num_batches_tracked]
    %layer2_1_bn3_running_mean : [num_users=1] = get_attr[target=layer2.1.bn3.running_mean]
    %layer2_1_bn3_running_var : [num_users=1] = get_attr[target=layer2.1.bn3.running_var]
    %layer2_1_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer2.1.bn3.num_batches_tracked]
    %layer2_2_bn1_running_mean : [num_users=1] = get_attr[target=layer2.2.bn1.running_mean]
    %layer2_2_bn1_running_var : [num_users=1] = get_attr[target=layer2.2.bn1.running_var]
    %layer2_2_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer2.2.bn1.num_batches_tracked]
    %layer2_2_bn2_running_mean : [num_users=1] = get_attr[target=layer2.2.bn2.running_mean]
    %layer2_2_bn2_running_var : [num_users=1] = get_attr[target=layer2.2.bn2.running_var]
    %layer2_2_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer2.2.bn2.num_batches_tracked]
    %layer2_2_bn3_running_mean : [num_users=1] = get_attr[target=layer2.2.bn3.running_mean]
    %layer2_2_bn3_running_var : [num_users=1] = get_attr[target=layer2.2.bn3.running_var]
    %layer2_2_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer2.2.bn3.num_batches_tracked]
    %layer2_3_bn1_running_mean : [num_users=1] = get_attr[target=layer2.3.bn1.running_mean]
    %layer2_3_bn1_running_var : [num_users=1] = get_attr[target=layer2.3.bn1.running_var]
    %layer2_3_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer2.3.bn1.num_batches_tracked]
    %layer2_3_bn2_running_mean : [num_users=1] = get_attr[target=layer2.3.bn2.running_mean]
    %layer2_3_bn2_running_var : [num_users=1] = get_attr[target=layer2.3.bn2.running_var]
    %layer2_3_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer2.3.bn2.num_batches_tracked]
    %layer2_3_bn3_running_mean : [num_users=1] = get_attr[target=layer2.3.bn3.running_mean]
    %layer2_3_bn3_running_var : [num_users=1] = get_attr[target=layer2.3.bn3.running_var]
    %layer2_3_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer2.3.bn3.num_batches_tracked]
    %layer3_0_bn1_running_mean : [num_users=1] = get_attr[target=layer3.0.bn1.running_mean]
    %layer3_0_bn1_running_var : [num_users=1] = get_attr[target=layer3.0.bn1.running_var]
    %layer3_0_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer3.0.bn1.num_batches_tracked]
    %layer3_0_bn2_running_mean : [num_users=1] = get_attr[target=layer3.0.bn2.running_mean]
    %layer3_0_bn2_running_var : [num_users=1] = get_attr[target=layer3.0.bn2.running_var]
    %layer3_0_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer3.0.bn2.num_batches_tracked]
    %layer3_0_bn3_running_mean : [num_users=1] = get_attr[target=layer3.0.bn3.running_mean]
    %layer3_0_bn3_running_var : [num_users=1] = get_attr[target=layer3.0.bn3.running_var]
    %layer3_0_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer3.0.bn3.num_batches_tracked]
    %layer3_0_downsample_1_running_mean : [num_users=1] = get_attr[target=layer3.0.downsample.1.running_mean]
    %layer3_0_downsample_1_running_var : [num_users=1] = get_attr[target=layer3.0.downsample.1.running_var]
    %layer3_0_downsample_1_num_batches_tracked : [num_users=1] = get_attr[target=layer3.0.downsample.1.num_batches_tracked]
    %layer3_1_bn1_running_mean : [num_users=1] = get_attr[target=layer3.1.bn1.running_mean]
    %layer3_1_bn1_running_var : [num_users=1] = get_attr[target=layer3.1.bn1.running_var]
    %layer3_1_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer3.1.bn1.num_batches_tracked]
    %layer3_1_bn2_running_mean : [num_users=1] = get_attr[target=layer3.1.bn2.running_mean]
    %layer3_1_bn2_running_var : [num_users=1] = get_attr[target=layer3.1.bn2.running_var]
    %layer3_1_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer3.1.bn2.num_batches_tracked]
    %layer3_1_bn3_running_mean : [num_users=1] = get_attr[target=layer3.1.bn3.running_mean]
    %layer3_1_bn3_running_var : [num_users=1] = get_attr[target=layer3.1.bn3.running_var]
    %layer3_1_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer3.1.bn3.num_batches_tracked]
    %layer3_2_bn1_running_mean : [num_users=1] = get_attr[target=layer3.2.bn1.running_mean]
    %layer3_2_bn1_running_var : [num_users=1] = get_attr[target=layer3.2.bn1.running_var]
    %layer3_2_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer3.2.bn1.num_batches_tracked]
    %layer3_2_bn2_running_mean : [num_users=1] = get_attr[target=layer3.2.bn2.running_mean]
    %layer3_2_bn2_running_var : [num_users=1] = get_attr[target=layer3.2.bn2.running_var]
    %layer3_2_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer3.2.bn2.num_batches_tracked]
    %layer3_2_bn3_running_mean : [num_users=1] = get_attr[target=layer3.2.bn3.running_mean]
    %layer3_2_bn3_running_var : [num_users=1] = get_attr[target=layer3.2.bn3.running_var]
    %layer3_2_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer3.2.bn3.num_batches_tracked]
    %layer3_3_bn1_running_mean : [num_users=1] = get_attr[target=layer3.3.bn1.running_mean]
    %layer3_3_bn1_running_var : [num_users=1] = get_attr[target=layer3.3.bn1.running_var]
    %layer3_3_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer3.3.bn1.num_batches_tracked]
    %layer3_3_bn2_running_mean : [num_users=1] = get_attr[target=layer3.3.bn2.running_mean]
    %layer3_3_bn2_running_var : [num_users=1] = get_attr[target=layer3.3.bn2.running_var]
    %layer3_3_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer3.3.bn2.num_batches_tracked]
    %layer3_3_bn3_running_mean : [num_users=1] = get_attr[target=layer3.3.bn3.running_mean]
    %layer3_3_bn3_running_var : [num_users=1] = get_attr[target=layer3.3.bn3.running_var]
    %layer3_3_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer3.3.bn3.num_batches_tracked]
    %layer3_4_bn1_running_mean : [num_users=1] = get_attr[target=layer3.4.bn1.running_mean]
    %layer3_4_bn1_running_var : [num_users=1] = get_attr[target=layer3.4.bn1.running_var]
    %layer3_4_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer3.4.bn1.num_batches_tracked]
    %layer3_4_bn2_running_mean : [num_users=1] = get_attr[target=layer3.4.bn2.running_mean]
    %layer3_4_bn2_running_var : [num_users=1] = get_attr[target=layer3.4.bn2.running_var]
    %layer3_4_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer3.4.bn2.num_batches_tracked]
    %layer3_4_bn3_running_mean : [num_users=1] = get_attr[target=layer3.4.bn3.running_mean]
    %layer3_4_bn3_running_var : [num_users=1] = get_attr[target=layer3.4.bn3.running_var]
    %layer3_4_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer3.4.bn3.num_batches_tracked]
    %layer3_5_bn1_running_mean : [num_users=1] = get_attr[target=layer3.5.bn1.running_mean]
    %layer3_5_bn1_running_var : [num_users=1] = get_attr[target=layer3.5.bn1.running_var]
    %layer3_5_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer3.5.bn1.num_batches_tracked]
    %layer3_5_bn2_running_mean : [num_users=1] = get_attr[target=layer3.5.bn2.running_mean]
    %layer3_5_bn2_running_var : [num_users=1] = get_attr[target=layer3.5.bn2.running_var]
    %layer3_5_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer3.5.bn2.num_batches_tracked]
    %layer3_5_bn3_running_mean : [num_users=1] = get_attr[target=layer3.5.bn3.running_mean]
    %layer3_5_bn3_running_var : [num_users=1] = get_attr[target=layer3.5.bn3.running_var]
    %layer3_5_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer3.5.bn3.num_batches_tracked]
    %layer4_0_bn1_running_mean : [num_users=1] = get_attr[target=layer4.0.bn1.running_mean]
    %layer4_0_bn1_running_var : [num_users=1] = get_attr[target=layer4.0.bn1.running_var]
    %layer4_0_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer4.0.bn1.num_batches_tracked]
    %layer4_0_bn2_running_mean : [num_users=1] = get_attr[target=layer4.0.bn2.running_mean]
    %layer4_0_bn2_running_var : [num_users=1] = get_attr[target=layer4.0.bn2.running_var]
    %layer4_0_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer4.0.bn2.num_batches_tracked]
    %layer4_0_bn3_running_mean : [num_users=1] = get_attr[target=layer4.0.bn3.running_mean]
    %layer4_0_bn3_running_var : [num_users=1] = get_attr[target=layer4.0.bn3.running_var]
    %layer4_0_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer4.0.bn3.num_batches_tracked]
    %layer4_0_downsample_1_running_mean : [num_users=1] = get_attr[target=layer4.0.downsample.1.running_mean]
    %layer4_0_downsample_1_running_var : [num_users=1] = get_attr[target=layer4.0.downsample.1.running_var]
    %layer4_0_downsample_1_num_batches_tracked : [num_users=1] = get_attr[target=layer4.0.downsample.1.num_batches_tracked]
    %layer4_1_bn1_running_mean : [num_users=1] = get_attr[target=layer4.1.bn1.running_mean]
    %layer4_1_bn1_running_var : [num_users=1] = get_attr[target=layer4.1.bn1.running_var]
    %layer4_1_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer4.1.bn1.num_batches_tracked]
    %layer4_1_bn2_running_mean : [num_users=1] = get_attr[target=layer4.1.bn2.running_mean]
    %layer4_1_bn2_running_var : [num_users=1] = get_attr[target=layer4.1.bn2.running_var]
    %layer4_1_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer4.1.bn2.num_batches_tracked]
    %layer4_1_bn3_running_mean : [num_users=1] = get_attr[target=layer4.1.bn3.running_mean]
    %layer4_1_bn3_running_var : [num_users=1] = get_attr[target=layer4.1.bn3.running_var]
    %layer4_1_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer4.1.bn3.num_batches_tracked]
    %layer4_2_bn1_running_mean : [num_users=1] = get_attr[target=layer4.2.bn1.running_mean]
    %layer4_2_bn1_running_var : [num_users=1] = get_attr[target=layer4.2.bn1.running_var]
    %layer4_2_bn1_num_batches_tracked : [num_users=1] = get_attr[target=layer4.2.bn1.num_batches_tracked]
    %layer4_2_bn2_running_mean : [num_users=1] = get_attr[target=layer4.2.bn2.running_mean]
    %layer4_2_bn2_running_var : [num_users=1] = get_attr[target=layer4.2.bn2.running_var]
    %layer4_2_bn2_num_batches_tracked : [num_users=1] = get_attr[target=layer4.2.bn2.num_batches_tracked]
    %layer4_2_bn3_running_mean : [num_users=1] = get_attr[target=layer4.2.bn3.running_mean]
    %layer4_2_bn3_running_var : [num_users=1] = get_attr[target=layer4.2.bn3.running_var]
    %layer4_2_bn3_num_batches_tracked : [num_users=1] = get_attr[target=layer4.2.bn3.num_batches_tracked]
    %x : [num_users=1] = placeholder[target=x]
    %conv2d : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%x, %conv1_weight, None, [2, 2], [3, 3]), kwargs = {})
    %add_ : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d, %bn1_weight, %bn1_bias, %bn1_running_mean, %bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu_ : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm,), kwargs = {})
    %max_pool2d : [num_users=2] = call_function[target=torch.ops.aten.max_pool2d.default](args = (%relu_, [3, 3], [2, 2], [1, 1]), kwargs = {})
    %conv2d_1 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%max_pool2d, %layer1_0_conv1_weight), kwargs = {})
    %add__1 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_0_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_1 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_1, %layer1_0_bn1_weight, %layer1_0_bn1_bias, %layer1_0_bn1_running_mean, %layer1_0_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__1 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_1,), kwargs = {})
    %conv2d_2 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__1, %layer1_0_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__2 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_0_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_2 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_2, %layer1_0_bn2_weight, %layer1_0_bn2_bias, %layer1_0_bn2_running_mean, %layer1_0_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__2 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_2,), kwargs = {})
    %conv2d_3 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__2, %layer1_0_conv3_weight), kwargs = {})
    %add__3 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_0_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_3 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_3, %layer1_0_bn3_weight, %layer1_0_bn3_bias, %layer1_0_bn3_running_mean, %layer1_0_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %conv2d_4 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%max_pool2d, %layer1_0_downsample_0_weight), kwargs = {})
    %add__4 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_0_downsample_1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_4 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_4, %layer1_0_downsample_1_weight, %layer1_0_downsample_1_bias, %layer1_0_downsample_1_running_mean, %layer1_0_downsample_1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__5 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_3, %batch_norm_4), kwargs = {})
    %relu__3 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__5,), kwargs = {})
    %conv2d_5 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__3, %layer1_1_conv1_weight), kwargs = {})
    %add__6 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_1_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_5 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_5, %layer1_1_bn1_weight, %layer1_1_bn1_bias, %layer1_1_bn1_running_mean, %layer1_1_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__4 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_5,), kwargs = {})
    %conv2d_6 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__4, %layer1_1_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__7 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_1_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_6 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_6, %layer1_1_bn2_weight, %layer1_1_bn2_bias, %layer1_1_bn2_running_mean, %layer1_1_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__5 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_6,), kwargs = {})
    %conv2d_7 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__5, %layer1_1_conv3_weight), kwargs = {})
    %add__8 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_1_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_7 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_7, %layer1_1_bn3_weight, %layer1_1_bn3_bias, %layer1_1_bn3_running_mean, %layer1_1_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__9 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_7, %relu__3), kwargs = {})
    %relu__6 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__9,), kwargs = {})
    %conv2d_8 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__6, %layer1_2_conv1_weight), kwargs = {})
    %add__10 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_2_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_8 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_8, %layer1_2_bn1_weight, %layer1_2_bn1_bias, %layer1_2_bn1_running_mean, %layer1_2_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__7 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_8,), kwargs = {})
    %conv2d_9 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__7, %layer1_2_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__11 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_2_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_9 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_9, %layer1_2_bn2_weight, %layer1_2_bn2_bias, %layer1_2_bn2_running_mean, %layer1_2_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__8 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_9,), kwargs = {})
    %conv2d_10 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__8, %layer1_2_conv3_weight), kwargs = {})
    %add__12 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer1_2_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_10 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_10, %layer1_2_bn3_weight, %layer1_2_bn3_bias, %layer1_2_bn3_running_mean, %layer1_2_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__13 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_10, %relu__6), kwargs = {})
    %relu__9 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__13,), kwargs = {})
    %conv2d_11 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__9, %layer2_0_conv1_weight), kwargs = {})
    %add__14 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_0_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_11 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_11, %layer2_0_bn1_weight, %layer2_0_bn1_bias, %layer2_0_bn1_running_mean, %layer2_0_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__10 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_11,), kwargs = {})
    %conv2d_12 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__10, %layer2_0_conv2_weight, None, [2, 2], [1, 1]), kwargs = {})
    %add__15 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_0_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_12 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_12, %layer2_0_bn2_weight, %layer2_0_bn2_bias, %layer2_0_bn2_running_mean, %layer2_0_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__11 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_12,), kwargs = {})
    %conv2d_13 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__11, %layer2_0_conv3_weight), kwargs = {})
    %add__16 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_0_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_13 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_13, %layer2_0_bn3_weight, %layer2_0_bn3_bias, %layer2_0_bn3_running_mean, %layer2_0_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %conv2d_14 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__9, %layer2_0_downsample_0_weight, None, [2, 2]), kwargs = {})
    %add__17 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_0_downsample_1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_14 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_14, %layer2_0_downsample_1_weight, %layer2_0_downsample_1_bias, %layer2_0_downsample_1_running_mean, %layer2_0_downsample_1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__18 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_13, %batch_norm_14), kwargs = {})
    %relu__12 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__18,), kwargs = {})
    %conv2d_15 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__12, %layer2_1_conv1_weight), kwargs = {})
    %add__19 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_1_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_15 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_15, %layer2_1_bn1_weight, %layer2_1_bn1_bias, %layer2_1_bn1_running_mean, %layer2_1_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__13 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_15,), kwargs = {})
    %conv2d_16 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__13, %layer2_1_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__20 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_1_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_16 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_16, %layer2_1_bn2_weight, %layer2_1_bn2_bias, %layer2_1_bn2_running_mean, %layer2_1_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__14 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_16,), kwargs = {})
    %conv2d_17 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__14, %layer2_1_conv3_weight), kwargs = {})
    %add__21 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_1_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_17 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_17, %layer2_1_bn3_weight, %layer2_1_bn3_bias, %layer2_1_bn3_running_mean, %layer2_1_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__22 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_17, %relu__12), kwargs = {})
    %relu__15 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__22,), kwargs = {})
    %conv2d_18 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__15, %layer2_2_conv1_weight), kwargs = {})
    %add__23 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_2_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_18 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_18, %layer2_2_bn1_weight, %layer2_2_bn1_bias, %layer2_2_bn1_running_mean, %layer2_2_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__16 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_18,), kwargs = {})
    %conv2d_19 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__16, %layer2_2_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__24 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_2_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_19 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_19, %layer2_2_bn2_weight, %layer2_2_bn2_bias, %layer2_2_bn2_running_mean, %layer2_2_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__17 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_19,), kwargs = {})
    %conv2d_20 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__17, %layer2_2_conv3_weight), kwargs = {})
    %add__25 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_2_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_20 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_20, %layer2_2_bn3_weight, %layer2_2_bn3_bias, %layer2_2_bn3_running_mean, %layer2_2_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__26 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_20, %relu__15), kwargs = {})
    %relu__18 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__26,), kwargs = {})
    %conv2d_21 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__18, %layer2_3_conv1_weight), kwargs = {})
    %add__27 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_3_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_21 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_21, %layer2_3_bn1_weight, %layer2_3_bn1_bias, %layer2_3_bn1_running_mean, %layer2_3_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__19 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_21,), kwargs = {})
    %conv2d_22 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__19, %layer2_3_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__28 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_3_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_22 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_22, %layer2_3_bn2_weight, %layer2_3_bn2_bias, %layer2_3_bn2_running_mean, %layer2_3_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__20 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_22,), kwargs = {})
    %conv2d_23 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__20, %layer2_3_conv3_weight), kwargs = {})
    %add__29 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer2_3_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_23 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_23, %layer2_3_bn3_weight, %layer2_3_bn3_bias, %layer2_3_bn3_running_mean, %layer2_3_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__30 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_23, %relu__18), kwargs = {})
    %relu__21 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__30,), kwargs = {})
    %conv2d_24 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__21, %layer3_0_conv1_weight), kwargs = {})
    %add__31 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_0_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_24 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_24, %layer3_0_bn1_weight, %layer3_0_bn1_bias, %layer3_0_bn1_running_mean, %layer3_0_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__22 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_24,), kwargs = {})
    %conv2d_25 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__22, %layer3_0_conv2_weight, None, [2, 2], [1, 1]), kwargs = {})
    %add__32 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_0_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_25 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_25, %layer3_0_bn2_weight, %layer3_0_bn2_bias, %layer3_0_bn2_running_mean, %layer3_0_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__23 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_25,), kwargs = {})
    %conv2d_26 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__23, %layer3_0_conv3_weight), kwargs = {})
    %add__33 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_0_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_26 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_26, %layer3_0_bn3_weight, %layer3_0_bn3_bias, %layer3_0_bn3_running_mean, %layer3_0_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %conv2d_27 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__21, %layer3_0_downsample_0_weight, None, [2, 2]), kwargs = {})
    %add__34 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_0_downsample_1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_27 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_27, %layer3_0_downsample_1_weight, %layer3_0_downsample_1_bias, %layer3_0_downsample_1_running_mean, %layer3_0_downsample_1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__35 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_26, %batch_norm_27), kwargs = {})
    %relu__24 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__35,), kwargs = {})
    %conv2d_28 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__24, %layer3_1_conv1_weight), kwargs = {})
    %add__36 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_1_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_28 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_28, %layer3_1_bn1_weight, %layer3_1_bn1_bias, %layer3_1_bn1_running_mean, %layer3_1_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__25 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_28,), kwargs = {})
    %conv2d_29 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__25, %layer3_1_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__37 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_1_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_29 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_29, %layer3_1_bn2_weight, %layer3_1_bn2_bias, %layer3_1_bn2_running_mean, %layer3_1_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__26 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_29,), kwargs = {})
    %conv2d_30 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__26, %layer3_1_conv3_weight), kwargs = {})
    %add__38 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_1_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_30 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_30, %layer3_1_bn3_weight, %layer3_1_bn3_bias, %layer3_1_bn3_running_mean, %layer3_1_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__39 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_30, %relu__24), kwargs = {})
    %relu__27 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__39,), kwargs = {})
    %conv2d_31 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__27, %layer3_2_conv1_weight), kwargs = {})
    %add__40 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_2_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_31 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_31, %layer3_2_bn1_weight, %layer3_2_bn1_bias, %layer3_2_bn1_running_mean, %layer3_2_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__28 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_31,), kwargs = {})
    %conv2d_32 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__28, %layer3_2_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__41 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_2_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_32 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_32, %layer3_2_bn2_weight, %layer3_2_bn2_bias, %layer3_2_bn2_running_mean, %layer3_2_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__29 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_32,), kwargs = {})
    %conv2d_33 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__29, %layer3_2_conv3_weight), kwargs = {})
    %add__42 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_2_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_33 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_33, %layer3_2_bn3_weight, %layer3_2_bn3_bias, %layer3_2_bn3_running_mean, %layer3_2_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__43 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_33, %relu__27), kwargs = {})
    %relu__30 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__43,), kwargs = {})
    %conv2d_34 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__30, %layer3_3_conv1_weight), kwargs = {})
    %add__44 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_3_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_34 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_34, %layer3_3_bn1_weight, %layer3_3_bn1_bias, %layer3_3_bn1_running_mean, %layer3_3_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__31 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_34,), kwargs = {})
    %conv2d_35 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__31, %layer3_3_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__45 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_3_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_35 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_35, %layer3_3_bn2_weight, %layer3_3_bn2_bias, %layer3_3_bn2_running_mean, %layer3_3_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__32 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_35,), kwargs = {})
    %conv2d_36 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__32, %layer3_3_conv3_weight), kwargs = {})
    %add__46 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_3_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_36 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_36, %layer3_3_bn3_weight, %layer3_3_bn3_bias, %layer3_3_bn3_running_mean, %layer3_3_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__47 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_36, %relu__30), kwargs = {})
    %relu__33 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__47,), kwargs = {})
    %conv2d_37 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__33, %layer3_4_conv1_weight), kwargs = {})
    %add__48 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_4_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_37 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_37, %layer3_4_bn1_weight, %layer3_4_bn1_bias, %layer3_4_bn1_running_mean, %layer3_4_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__34 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_37,), kwargs = {})
    %conv2d_38 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__34, %layer3_4_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__49 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_4_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_38 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_38, %layer3_4_bn2_weight, %layer3_4_bn2_bias, %layer3_4_bn2_running_mean, %layer3_4_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__35 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_38,), kwargs = {})
    %conv2d_39 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__35, %layer3_4_conv3_weight), kwargs = {})
    %add__50 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_4_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_39 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_39, %layer3_4_bn3_weight, %layer3_4_bn3_bias, %layer3_4_bn3_running_mean, %layer3_4_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__51 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_39, %relu__33), kwargs = {})
    %relu__36 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__51,), kwargs = {})
    %conv2d_40 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__36, %layer3_5_conv1_weight), kwargs = {})
    %add__52 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_5_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_40 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_40, %layer3_5_bn1_weight, %layer3_5_bn1_bias, %layer3_5_bn1_running_mean, %layer3_5_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__37 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_40,), kwargs = {})
    %conv2d_41 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__37, %layer3_5_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__53 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_5_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_41 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_41, %layer3_5_bn2_weight, %layer3_5_bn2_bias, %layer3_5_bn2_running_mean, %layer3_5_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__38 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_41,), kwargs = {})
    %conv2d_42 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__38, %layer3_5_conv3_weight), kwargs = {})
    %add__54 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer3_5_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_42 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_42, %layer3_5_bn3_weight, %layer3_5_bn3_bias, %layer3_5_bn3_running_mean, %layer3_5_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__55 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_42, %relu__36), kwargs = {})
    %relu__39 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__55,), kwargs = {})
    %conv2d_43 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__39, %layer4_0_conv1_weight), kwargs = {})
    %add__56 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_0_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_43 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_43, %layer4_0_bn1_weight, %layer4_0_bn1_bias, %layer4_0_bn1_running_mean, %layer4_0_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__40 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_43,), kwargs = {})
    %conv2d_44 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__40, %layer4_0_conv2_weight, None, [2, 2], [1, 1]), kwargs = {})
    %add__57 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_0_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_44 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_44, %layer4_0_bn2_weight, %layer4_0_bn2_bias, %layer4_0_bn2_running_mean, %layer4_0_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__41 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_44,), kwargs = {})
    %conv2d_45 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__41, %layer4_0_conv3_weight), kwargs = {})
    %add__58 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_0_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_45 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_45, %layer4_0_bn3_weight, %layer4_0_bn3_bias, %layer4_0_bn3_running_mean, %layer4_0_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %conv2d_46 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__39, %layer4_0_downsample_0_weight, None, [2, 2]), kwargs = {})
    %add__59 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_0_downsample_1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_46 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_46, %layer4_0_downsample_1_weight, %layer4_0_downsample_1_bias, %layer4_0_downsample_1_running_mean, %layer4_0_downsample_1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__60 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_45, %batch_norm_46), kwargs = {})
    %relu__42 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__60,), kwargs = {})
    %conv2d_47 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__42, %layer4_1_conv1_weight), kwargs = {})
    %add__61 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_1_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_47 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_47, %layer4_1_bn1_weight, %layer4_1_bn1_bias, %layer4_1_bn1_running_mean, %layer4_1_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__43 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_47,), kwargs = {})
    %conv2d_48 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__43, %layer4_1_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__62 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_1_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_48 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_48, %layer4_1_bn2_weight, %layer4_1_bn2_bias, %layer4_1_bn2_running_mean, %layer4_1_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__44 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_48,), kwargs = {})
    %conv2d_49 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__44, %layer4_1_conv3_weight), kwargs = {})
    %add__63 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_1_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_49 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_49, %layer4_1_bn3_weight, %layer4_1_bn3_bias, %layer4_1_bn3_running_mean, %layer4_1_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__64 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_49, %relu__42), kwargs = {})
    %relu__45 : [num_users=2] = call_function[target=torch.ops.aten.relu_.default](args = (%add__64,), kwargs = {})
    %conv2d_50 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__45, %layer4_2_conv1_weight), kwargs = {})
    %add__65 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_2_bn1_num_batches_tracked, 1), kwargs = {})
    %batch_norm_50 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_50, %layer4_2_bn1_weight, %layer4_2_bn1_bias, %layer4_2_bn1_running_mean, %layer4_2_bn1_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__46 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_50,), kwargs = {})
    %conv2d_51 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__46, %layer4_2_conv2_weight, None, [1, 1], [1, 1]), kwargs = {})
    %add__66 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_2_bn2_num_batches_tracked, 1), kwargs = {})
    %batch_norm_51 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_51, %layer4_2_bn2_weight, %layer4_2_bn2_bias, %layer4_2_bn2_running_mean, %layer4_2_bn2_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %relu__47 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%batch_norm_51,), kwargs = {})
    %conv2d_52 : [num_users=1] = call_function[target=torch.ops.aten.conv2d.default](args = (%relu__47, %layer4_2_conv3_weight), kwargs = {})
    %add__67 : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%layer4_2_bn3_num_batches_tracked, 1), kwargs = {})
    %batch_norm_52 : [num_users=1] = call_function[target=torch.ops.aten.batch_norm.default](args = (%conv2d_52, %layer4_2_bn3_weight, %layer4_2_bn3_bias, %layer4_2_bn3_running_mean, %layer4_2_bn3_running_var, True, 0.1, 1e-05, True), kwargs = {})
    %add__68 : [num_users=1] = call_function[target=torch.ops.aten.add_.Tensor](args = (%batch_norm_52, %relu__45), kwargs = {})
    %relu__48 : [num_users=1] = call_function[target=torch.ops.aten.relu_.default](args = (%add__68,), kwargs = {})
    %adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.adaptive_avg_pool2d.default](args = (%relu__48, [1, 1]), kwargs = {})
    %flatten : [num_users=1] = call_function[target=torch.ops.aten.flatten.using_ints](args = (%adaptive_avg_pool2d, 1), kwargs = {})
    %linear : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%flatten, %fc_weight, %fc_bias), kwargs = {})
    return (linear,)
```
每行的第一个元素为 %node.name
目前只能通过人工观察，找到要配置的算子的 node.name，配置到 config 的 module_names 中
比如配置第2到第6个 conv 为 U16 量化，在 config 中配置 
`"module_names": ["conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5"],`
前两个 block 中还有 2 个 add，在 nodes 里找到 node.name，配置
`"module_names":["add__5", "add__9"]`

在上面分支下运行 `python3 -m resnet50.train` 可以保存混合精度量化的 resnet50

### 几点说明：
- module_names 为可选配置，module_type 为必须配置，为了支持 yolov5s，目前只支持了"add", "conv", "concat", "silu", "avgpool", "linear"算子的量化
- 量化配置实际上是以 pattern 为单位，而不是算子，比如上面 U16 量化的 conv 其实是 conv_bn_relu ，作为一组 pattern，暂时 module_type 只需要配置 pattern 中的核心计算算子，以后会添加 module_type 配置的详细说明。