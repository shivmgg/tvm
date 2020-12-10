import tvm
from tvm import relay
from tvm.contrib import util
from tvm.relay.op.contrib import nvdla

from itertools import zip_longest, combinations
import json
import os
import warnings

import numpy as np
from collections import OrderedDict 

data_type = "int8"
np.random.seed(42)

## LeNet Network
data_shape = (1, 1, 28, 28)
weight1_shape = (20, 1, 5, 5)
conv_bias1_shape = [20]

data = relay.var("data", shape=data_shape, dtype="int8")

conv1_weight = relay.var('conv2d_0', shape=weight1_shape, dtype=data_type)
conv1_bias = relay.var("bias_add_0", shape=conv_bias1_shape, dtype=data_type) 
conv1 = relay.nn.conv2d(data, conv1_weight, kernel_size=(5, 5), strides=(1, 1), padding=(0, 0), channels=20)
conv1 = relay.nn.bias_add(conv1, conv1_bias)
maxpool1 = relay.nn.max_pool2d(conv1, (2, 2), (2, 2))

weight2_shape = (50, 20, 5, 5)
conv_bias2_shape = [50]
conv2_weight = relay.var('conv2d_1', shape=weight2_shape, dtype=data_type)
conv2_bias = relay.var("bias_add_1", shape=conv_bias2_shape, dtype=data_type)
conv2 = relay.nn.conv2d(maxpool1, conv2_weight, kernel_size=(5, 5), strides=(1, 1), padding=(0, 0), channels=50)
conv2 = relay.nn.bias_add(conv2, conv2_bias)
maxpool2 = relay.nn.max_pool2d(conv2, (2, 2), (2, 2))

dense1_shape = (500, 800)
dense_bias1_shape = [500]
dense1_weight = relay.var('dense_1', shape=dense1_shape, dtype=data_type)
dense1_bias = relay.var("bias_add_2", shape=dense_bias1_shape, dtype=data_type) 

dense2_shape = (10, 500)
dense_bias2_shape = [10]
dense2_weight = relay.var('dense_2', shape=dense2_shape, dtype=data_type)
dense2_bias = relay.var("bias_add_3", shape=dense_bias2_shape, dtype=data_type) 

# Complete Network architecture
bf1 = relay.nn.batch_flatten(maxpool2)
dense1 = relay.nn.dense(bf1, dense1_weight, units=500)
dense1 = relay.nn.bias_add(dense1, dense1_bias)
dense1 = relay.nn.relu(dense1)
dense2 = relay.nn.dense(dense1, dense2_weight, units=10)
dense2 = relay.nn.bias_add(dense2, dense2_bias)
# softmax = relay.nn.softmax(dense2)
module = tvm.IRModule.from_expr(dense2)
mod = module

#######################################################################################################
import sys 
from PIL import Image
img = Image.open('examples/test_images/input0.pgm')
#img = Image.open('examples/test_images/0_3.jpg')
img_out = np.asarray(img)[None][None]

#Assigning weights to corresponding layers
low = -128
high = 128

#######################################################################################################
#Uncomment if loading weights and bias directly from JSON

# with open('/path_to_json/json/NVDLA_lenet.json') as f:
#   d = json.load(f)

# params = dict()
# weight_data = d["conv2d_0"]["weights"]
# params['conv2d_0'] = tvm.nd.array(np.asarray(weight_data).reshape(weight1_shape).astype(data_type))
# params['bias_add_0'] = tvm.nd.array(np.asarray(d["bias_add_0"]["bias"]).reshape(conv_bias1_shape).astype(data_type))
# params['conv2d_1'] = tvm.nd.array(np.asarray(d["conv2d_1"]["weights"]).reshape(weight2_shape).astype(data_type))
# params['bias_add_1'] = tvm.nd.array(np.asarray(d["bias_add_1"]["bias"]).reshape(conv_bias2_shape).astype(data_type))
# params['dense_1'] = tvm.nd.array(np.asarray(d["dense_1"]["weights"]).reshape(dense1_shape).astype(data_type))
# params['bias_add_2'] = tvm.nd.array(np.asarray(d["bias_add_2"]["bias"]).reshape(dense_bias1_shape).astype(data_type))
# params['dense_2'] = tvm.nd.array(np.asarray(d["dense_2"]["weights"]).reshape(dense2_shape).astype(data_type))
# params['bias_add_3'] = tvm.nd.array(np.asarray(d["bias_add_3"]["bias"]).reshape(dense_bias2_shape).astype(data_type))

##################################################################################################
# Initializing weights and biases
# Comment if loading from a JSON
data = tvm.nd.array(np.random.uniform(low, high, data_shape).astype(data_type))
weight1 = tvm.nd.array(np.random.uniform(low, high, weight1_shape).astype(data_type))
bias1 = tvm.nd.array(np.random.uniform(low, high, conv_bias1_shape).astype(data_type))

weight2 = tvm.nd.array(np.random.uniform(low, high, weight2_shape).astype(data_type))
bias2 = tvm.nd.array(np.random.uniform(low, high, conv_bias2_shape).astype(data_type))

weight3 = tvm.nd.array(np.random.uniform(low, high, dense1_shape).astype(data_type))
bias3 = tvm.nd.array(np.random.uniform(low, high, dense_bias1_shape).astype(data_type))

weight4 = tvm.nd.array(np.random.uniform(low, high, dense2_shape).astype(data_type))
bias4 = tvm.nd.array(np.random.uniform(low, high, dense_bias2_shape).astype(data_type))

params = dict()
params['conv2d_0'] = weight1
params['bias_add_0'] = bias1
params['conv2d_1'] = weight2
params['bias_add_1'] = bias2
params['dense_1'] = weight3
params['bias_add_2'] = bias3
params['dense_2'] = weight4
params['bias_add_3'] = bias4

#################################################################################
# Compiling the network using default LLVM compiler to obtain expected output
from tvm.contrib import graph_runtime
target = "llvm"

# Which device to run on. Should be one of tvm.cpu() or tvm.gpu().
ctx = tvm.cpu()

data = np.array(img_out)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    executor = relay.build_module.create_executor("graph", mod, ctx, target)    

dtype = "int8"
tvm_out = executor.evaluate()(data, **params)
tvm_output = tvm_out.asnumpy()

outp = dict()
outp['shape'] = tvm_output.shape
tvm_output = tvm_output.tolist()
outp['value'] = tvm_output
with open('/path_to_tvm/tvm/examples/expected_output/expected_output.json', 'w') as outfile:
    outfile.write(json.dumps(outp, sort_keys=True, indent=2))
print(tvm_output)

# # Rounding off output values
# tvm_output = np.where(np.around(np.array(tvm_output), 1) > 0, 1, np.where(np.around(np.array(tvm_output), 1) < 0, -1, 0))
# # print(tvm_output)
# tvm_output = tvm_output.tolist()
# outp['value'] = tvm_output
# with open('/home/shivam/NUS/tvm/final_json/rounded_output.json', 'w') as outfile:
#     outfile.write(json.dumps(outp, sort_keys=True, indent=2))

res = dict([key, value.asnumpy()]  
       for key, value in params.items())

def extract_nvdla_modules(module):
    """Get the NVDLA module(s) from llvm module."""
    return list(filter(lambda mod: mod.type_key == "nvdla",
                       module.get_lib().imported_modules))

target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
enable_nvdla = True
tvm_ops=0
nvdla_partitions=1

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    if enable_nvdla:
        module = nvdla.partition_for_nvdla(module, params)
    lib = relay.build(module, target=target, params=params)
    nvdla_modules = extract_nvdla_modules(lib)
    layers = dict()
    l = dict()
    for mod in nvdla_modules:
        source = mod.get_source("json")
        codegen = json.loads(source)["nodes"]
        input_name = codegen[0]['name']
        split_name = input_name.split('_')
        if (int(split_name[1]) == 0):
                l['input'] = codegen[0]['attrs']

        if len(codegen) == 2:
            layers[int(split_name[1])] = codegen[1]
        elif len(codegen) > 2:
            layers[int(split_name[1])] = codegen[2]
        codegen_str = json.dumps(codegen, sort_keys=True, indent=2)
    od = OrderedDict(sorted(layers.items()))

    q = 0
    iter = 0

    # Adding source and destination
    l['input']['id'] = 'node' + '_' + str(iter)
    l['input']['source'] = []
    l['input']['dest'] = ['node' + '_' + str(iter + 1)]

    iter = iter + 1
    for k, v in od.items(): 
        # Assigning name to a layer
        layer_name = v['name'].split('.')[1] + '_' + str(q)
        if (layer_name in l.keys()):
            q = q + 1
            layer_name = v['name'].split('.')[1] + '_' + str(q)
        v['attrs']['id'] = 'node' + '_' + str(iter)

        if iter != 0:
            v['attrs']['source'] = ['node' + '_' + str(iter - 1)]
        else:
            v['attrs']['source'] = []

        if iter == len(od.values()):
            v['attrs']['dest'] = []
        else:
            v['attrs']['dest'] = ['node' + '_' + str(iter + 1)]
        l[layer_name] = v['attrs']

        if layer_name in res.keys():
            if "bias" not in layer_name:
                l[layer_name]['weight_shape'] = list(res[layer_name].shape)
                l[layer_name]['weights'] = res[layer_name].flatten().tolist()
            if "bias" in layer_name:
                l[layer_name]['bias_shape'] = list(res[layer_name].shape)
                l[layer_name]['bias'] = res[layer_name].flatten().tolist()       
        iter = iter + 1

    with open('/path_to_tvm/tvm/examples/json/NVDLA_lenet.json', 'w') as outfile:
        json.dump(l, outfile)



