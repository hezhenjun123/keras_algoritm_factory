import os
import argparse
import numpy as np
import tensorflow as tf

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import uff
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def GiB(val):
    return val * 1 << 30

class ModelData(object):
    def __init__(self, pb_file_path, input_name, trt_input_shape, output_name, fp16_mode, trt_engine_path):
        self.pb_file_path = pb_file_path
        #self.uff_file_path = uff_file_path
        self.input_name = input_name
        self.input_shape = trt_input_shape
        self.output_name = output_name
        self.fp16_mode = fp16_mode
        self.trt_engine_path = trt_engine_path

def build_engine(model_data):
    uff_model = uff.from_tensorflow_frozen_model(model_data.pb_file_path)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = GiB(5)
        builder.fp16_mode = model_data.fp16_mode
        parser.register_input(model_data.input_name, model_data.input_shape)
        parser.register_output(model_data.output_name)
        parser.parse_buffer(uff_model, network)
        return builder.build_cuda_engine(network)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


class ModelInferTF(object):
    def __init__(self, model_filepath, input_name, output_name):

        self.model_filepath = model_filepath
        self.load_graph(model_filepath = self.model_filepath)
        self.input_name = input_name
        self.output_name = output_name

    def load_graph(self, model_filepath):
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self.graph.finalize()


        print('Model loading complete!')

        opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.0625)
        config = tf.ConfigProto(gpu_options=opts)

        self.sess = tf.Session(graph = self.graph, config=config)

    def infer(self, data):
        output_tensor = self.graph.get_tensor_by_name(self.output_name + ":0")
        start_time = time.time()
        output = self.sess.run([output_tensor], feed_dict = {self.input_name + ":0": data})
        #print("--- %s seconds ---" % (time.time() - start_time))
        return output
