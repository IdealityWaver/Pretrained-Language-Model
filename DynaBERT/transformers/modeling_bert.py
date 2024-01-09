# coding=utf-8
# 2020.08.28 - (1) Changed fixed sized Transformer layer to support adptive width; and
#              (2) Added modules to rewire the BERT model according to the importance of attention
#                  heads and neurons in the intermediate layer of Feed-forward Network.
#              Huawei Technologies Co., Ltd <houlu3@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys
import time
sys.path.append("..")

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np

from .modeling_utils import PreTrainedModel
from .configuration_bert import BertConfig

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
}


def detect_outliers(weights):
    from sklearn.mixture import GaussianMixture 
    print("detecting outliers...")
    # find outliers
    _weights = weights.reshape(-1, 1)
    gm = GaussianMixture(n_components=1, random_state=0).fit(_weights)
    scores = gm.score_samples(_weights)
    outliers = []
    _weights = _weights.numpy()
    for i in range(0, len(scores)):
        if scores[i] <= -4.0:
            outliers.append(_weights[i][0])
    print("masked: ", len(outliers))
    mask = np.zeros(weights.shape, dtype=bool)
    mask[np.where(scores <= -4.0)] = True
    return np.array(outliers), mask 


def gather_layer_weights(layer):
    weights = torch.Tensor([])
    # original dimension and pointer address to restore to
    orig = []
    for name, param in layer.named_parameters():
        if param.requires_grad:
            weights = weights.to(param.data.device)
            orig.append((param, name))
            weights = torch.cat([weights, torch.flatten(param.data)])
    return weights, orig 


# lwg: gobo implementation 
# returns quantized matrix
'''
@Author: liux
@Time: 2023/12/27 20:14:02
@Desc: o_idx为bool类型的array，weights为tensor，二者大小相同。
weughts对应位置值为True的代表该元素是异常值。
'''
def gobo_quantize(weights, o_idx, bits):
    print("gobo qunatization. Bits = ", bits)
    # liux: 将weights从tensor转变成numpy。因为o_idx是numpy，存储的元素类型位dtype('bool')，True为1，False为0
    # 用dtype('bool')的numpy对tensor做mask，实际上是tensor取下标0或者1的元素。
    weights = weights.numpy()
    # liux: 对weights做mask，取出正常值。
    g_group = weights[~o_idx]
    # liux: 对所有正常值从小到大排序。
    g_group = np.sort(g_group)
    # liux: 将正常值分成2^n组，每step个数分为1组，将一组的平均值作为该组的代表值，并取该组的第一个值为界线。
    bins = []
    n_bins = pow(2, bits)
    step = int(len(g_group)/n_bins)
    centroids = []
    # calculate centroids in G group
    for i in range(n_bins):
        start = i * step
        centroids.append(np.average(g_group[start: start + step]))
        bins.append(g_group[start])
    # liux: 界线加入最后一个数，即bins为[g_group[0],......,g_group[-1]]。
    bins.append(g_group[-1])
    # print("centroids boundary: ", bins)
    #centroids.append(-99999.0) 
    # liux: 这里为什么要append centroids[0]？g_group应该还剩下最后不足一个step的一小段没有处理，应该是append这最后一小段的average。
    # 如果没有最后一小段的剩余，即len(g_group)/n_bins能整除，那就不应该再加入下面这段
    # centroids.append(np.average(g_group[n_bins * step:]))
    centroids.append(centroids[0]) 
    # print("centroids : ", centroids)
    centroids = np.array(centroids)
    # assign quantized values
    # liux: 将每个权重都变成质心的idx，所有小于等于最小正常值的为-1（最小质心），所有大于最大正常值的为256（最小质心）
    # 如果right为True，那么weights中小于等于最小界线的值都是0，减1则为-1，-1在centroids中的下标是最后一个，因此会取到最后一个centroids。
    # 而最后一个centroids之前append了最小值。
    quantized = np.digitize(weights, bins, right = True) - 1 # return the idx of the centroids
    print("quantzied weights are:", quantized)
    #save_weight_to_file('/tmp/weight', quantized)
    # start = time.time()
    # liux: 生成新的权重，将每个值变成平均值。
    new_weights = centroids[quantized]
    # recover corresponding outlier weights
    # liux: 恢复新权重中的异常值。
    new_weights[o_idx] = weights[o_idx]
    # end = time.time()
    # print("restoring weight takes " ,(end-start) * 1000, "ms")
    print("centroids size: " , centroids.shape) 
    print("bins size: ", len(bins))
    print("quantized weight size: ", quantized.shape)
    print("original weight size: ", weights.size) 
    #print("outlier size: ", len(o_idx))

    # sanity check manually patch some boundary values...
    # for idx,d in enumerate(new_weights):
    #     if d < -100.0:
    #         print(idx, weights[idx],"fail to be binned?? new_weight =", new_weights[idx])
    #         # there are values that are not in outlier idx 
    #         if idx not in o_idx:
    #             new_weights[idx] = centroids[0]
    #             print("manually patch", weights[idx], "to", centroids[0])
    return new_weights

'''
@Author: liux
@Time: 2023/12/24 15:51:40
量化某一层。
'''
# lwg: unified quantization entry
# layer: layer to be quantized
# quantize_f: gobo/kmeans, returns quantized new weights 
# detect_o: whether to detect outliers  
# bits: quantizatio bits 
def _quantize(layer, quantize_f, detect_o=True, bits=3):
    # keep the same as numpy print
    torch.set_printoptions(precision=8)
    weights, orig_param = gather_layer_weights(layer)
    if detect_o:
        o_group, o_idx = detect_outliers(weights)
        print("o_group = %d" % (len(o_group)))
    else:
        o_idx = []
    new_weights = quantize_f(weights, o_idx, bits)

    # liux: 下面这段for循环似乎无意义。似乎是为了复原模型参数，但是和下面的重复了，并且new_weights在第二个复原代码中被置为空。
    # restore to original NN module
    # for (src, name) in orig_param:
    #     size = src.data.size()
    #     length = src.data.nelement()
    #     orig = new_weights[:length]
    #     new_weights = new_weights[length:]
    #     # only qunatize non-layernorm modules in a layer 
    #     if "LayerNorm" not in name:
    #         print("skipping other layer")
    #         continue
    #     src.data = torch.from_numpy(orig).float().view(size)

    o_count = 0
    start = time.time()
    # liux: 将量化后的参数应用到模型中。
    # apply quantized weights to original NN module
    for name, param in layer.named_parameters():
        if param.requires_grad:
            size = param.data.size()
            length = param.data.nelement()
            # unroll the parameters from the large quantized weights
            this_module = torch.from_numpy(new_weights[:length]).float().view(size)
            new_weights = new_weights[length:]
            ''' 
            # check mixed quantization choice for different modules within a layer
            if "LayerNorm" not in name:
                continue
            '''

            '''
            for m in layer.children():
                print(m)
            # mix quantized attention head
            original_heads = torch.split(param.data, 64, 0)
            this_heads = torch.split(this_module, 64, 0)
            print("module size:", param.data.size())
            print("head size: ", original_heads[0].size())
            #this_module = torch.cat((torch.cat((this_heads[:6])), torch.cat((original_heads[6:])))).view(size)
            # ----
            '''
            param.data = this_module
            #o_this = 0
            o_this = len(np.nonzero(np.in1d(param.data.flatten(), o_group))[0])
            o_count += o_this
            #print("outliers in", name, ":", "{:.2%}".format(o_this/param.data.nelement()))
    end = time.time()
    print("total outliers in this layer:", "{:.2%}".format(o_count/weights.nelement()))
    # the measured time is not meaningful as it is pytorch implementation
    print("time to restore compressed weights:", (end - start)*1000)
    # sanity check, all outliers must be preserved 
    assert o_count == len(o_group)
    return o_count


def save_weight_to_file(filename, weights):
    f = open(filename, 'wb')
    np.save(f, weights)

# data: all weights of a Bert layer, in torch Tensor format
# bits: decides # of bins = 2^bits
# XXX: this is GOBO base without L1 Norm error minimization 
# XXX: accuracy is really bad without outlier detection
# XXX: bewlow obsolete -- look at _quantize
def gobo_quantize_one_layer(layer, bits=3):
    # keep the same as numpy print
    torch.set_printoptions(precision=8)
    # gather all weights from a layer
    weights, ctx = gather_layer_weights(layer)
    print(ctx)
    outliers, _ = detect_outliers(weights)
    #outliers = []
    print("torch weights:", weights)
    # tensor(-0.1389).numpy() gives -0.13887332 because torch print precision is 4 < 8 of numpy 
    # print(torch.flatten(weights)[199396].numpy())
    weights = weights.numpy()
    print("numpy weights:", weights)
    masked_weights = np.ma.MaskedArray(weights, np.in1d(weights, outliers))
    g_group = masked_weights[~masked_weights.mask]
    weights = np.sort(g_group)
    bins = []
    n_bins = pow(2, bits)
    step = int(len(weights)/n_bins)
    centroids = []
    # calculate centroids in G group
    for i in range(n_bins):
        start = i * step
        centroids.append(np.average(weights[start: start + step]))
        bins.append(weights[start])
    # last value for all outliers
    bins.append(weights[-1])
    bins = np.array(bins, dtype=float)
    print("bins are: ", bins)
    # for outliers 
    centroids.append(-99999.0)
    centroids = np.array(centroids)
    print("centroids are: ", centroids)
    # quantize by centroids 
    o_count = 0
    for name, param in layer.named_parameters():
        if param.requires_grad:
            old_size = param.data.size()
            data = torch.flatten(param.data).numpy()
            orig = torch.flatten(param.data).numpy()
            # save idx and weights of outliers in this NN module
            outliers_idx = np.nonzero(np.in1d(data, outliers))
            outliers_weights = data[outliers_idx]
            o_count += len(outliers_weights)
            print("outlier in this layer have:", len(outliers_weights))
            quantized = np.digitize(data, bins, right = True) - 1 # return the idx of the centroids
            print("quantized matrix is:", quantized)
            for idx,v in enumerate(quantized):
                if v == len(centroids):
                    if data[idx] not in outliers:
                        print("not in outlier...")
                        print(idx)
                        print(data[idx])
            # assign centroids 
            data = centroids[quantized]
            # recover corresponding weights
            data[outliers_idx] = outliers_weights 
            # why still unpatched value??
            for idx,d in enumerate(data):
                if d < -100.0:
                    print(idx, orig[idx],"??")
                    print(weights)
                    # there are values that are not in outlier idx 
                    if orig[idx] not in outliers:
                        data[idx] = centroids[0]
                        print("manually patch", orig[idx], "to", data[idx])
            data = torch.from_numpy(data).float()
            data = data.view(old_size)
            param.data = data
            print("after size:", param.data.size())
            print(param.data)
    assert o_count == len(outliers)
    print("recovered ", o_count, "outliers for current layer")


# lwg:to mearge w/ gobo
# XXX: bewlow obsolete -- look at _quantize
def kmeans_quantize_one_layer(layer, bits):
    import numpy as np
    from sklearn.cluster import KMeans
    # gather all weights from a layer
    weights, _ = gather_layer_weights(layer)
    outliers, _ = detect_outliers(weights)
    weights = weights.numpy()
    masked_weights = np.ma.MaskedArray(weights, np.in1d(weights, outliers))
    g_group = torch.from_numpy(masked_weights[~masked_weights.mask].reshape(-1, 1))
    km = KMeans(n_clusters=pow(2, bits), random_state=0, tol=1e-8).fit(g_group)
    print(km.labels_)
    print(km.cluster_centers_)
    for name, param in layer.named_parameters():
        if param.requires_grad:
            old_size = param.data.size()
            data = torch.flatten(param.data).numpy()
            param.data = torch.flatten(param.data).reshape(-1,1)
            outliers_idx = np.nonzero(np.in1d(data, outliers))
            outliers_weights = data[outliers_idx]
            print(outliers_idx)
            print(outliers_weights)
            labels = km.predict(param.data)
            print(labels)
            new_param = km.cluster_centers_[labels].flatten()
            print("quantized:", new_param)
            idx = outliers_idx[0].flatten()
            new_param[idx] = outliers_weights
            param.data = torch.from_numpy(new_param).float().view(old_size)
            print(param.data.size())
    
def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}
BertLayerNorm = torch.nn.LayerNorm


def round_to_nearest(input_size, width_mult, num_heads, min_value=1):
    new_width_mult = round(num_heads * width_mult)*1.0/num_heads
    input_size = int(new_width_mult * input_size)
    new_input_size = max(min_value, input_size)
    return new_input_size


'''
@Author: liux
@Time: 2023/12/30 14:35:34
@Desc: 用来缩减模型宽度，dyna_dim为长度为2的bool数组，第一个bool代表是否只计算一部分输入特征，第二个bool代表是否只输出一部分特征。
默认情况下是不缩减，和普通的nn.Linear一样的功能。
'''
class DynaLinear(nn.Linear):
    # liux: dyna_dim如果为[False, True]表示输入特征不变，输出特征缩减。相邻两个DynaLinear的dyna_dim应该在边界处保持一致。
    def __init__(self, in_features, out_features, num_heads, bias=True, dyna_dim=[True, True]):
        super(DynaLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.num_heads = num_heads
        self.width_mult = 1.
        self.dyna_dim = dyna_dim

    def forward(self, input):
        if self.dyna_dim[0]:
            self.in_features = round_to_nearest(self.in_features_max, self.width_mult, self.num_heads)
        if self.dyna_dim[1]:
            self.out_features = round_to_nearest(self.out_features_max, self.width_mult, self.num_heads)
        weight = self.weight[:self.out_features, :self.in_features]
        # print("in {0}, out {1}".format(self.in_features, self.out_features))
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def quantize(self, bits):
        print("------------------quantize word_embeddings------------------")
        _quantize(self.word_embeddings, gobo_quantize, detect_o=True, bits=bits)
        print("------------------quantize position_embeddings------------------")
        _quantize(self.position_embeddings, gobo_quantize, detect_o=True, bits=bits)
        print("------------------quantize token_type_embeddings------------------")
        _quantize(self.token_type_embeddings, gobo_quantize, detect_o=True, bits=bits)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print("embeddings for ", input_ids, ": ", embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.orig_num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # liux: 所有head在一个dense中。
        # dense layer for adaptive width
        self.query = DynaLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[False, True])
        self.key = DynaLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[False, True])
        self.value = DynaLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[False, True])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        self.num_attention_heads = round(self.orig_num_attention_heads * self.query.width_mult)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # liux: mixed_query_layer[bs, L, all_head_size] -> query_layer[bs, head_num, L, head_size]，key和value也是同理。
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # liux: attention_scores[bs, head_num, L, L]
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # liux: context_layer[bs, head_num, L, head_size]
        context_layer = torch.matmul(attention_probs, value_layer)

        # liux: context_layer[bs, head_num, L, head_size] -> context_layer[bs, L, head_num, head_size] -> context_layer[bs, L, all_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # output attention scores when needed
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        # dense layer for adaptive width
        self.dense = DynaLinear(config.hidden_size, config.hidden_size, config.num_attention_heads, dyna_dim=[True, False])
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # liux: Bert源码中norm是input_tensor+norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        # liux: 存储不同量化位宽的模块。
        # models with other bit widths of the same layer
        self.quantized = {}
    
    # lwg: idx is a list of head idx (0-11), sorted by their importance in descending order
    def reorder_heads(self, idx):
        n, a = self.self.num_attention_heads, self.self.attention_head_size
        # group every 64-dim as an attention head and reorder as described in idx array
        index = torch.arange(n*a).reshape(n, a)[idx].view(-1).contiguous().long()

        def reorder_head_matrix(linearLayer, index, dim=0):
            index = index.to(linearLayer.weight.device)
            W = linearLayer.weight.index_select(dim, index).clone().detach()
            if linearLayer.bias is not None:
                # if dim == 1, then do not reorder bias
                if dim == 1:
                    b = linearLayer.bias.clone().detach()
                else:
                    b = linearLayer.bias[index].clone().detach()

            linearLayer.weight.requires_grad = False
            linearLayer.weight.copy_(W.contiguous())
            linearLayer.weight.requires_grad = True
            if linearLayer.bias is not None:
                linearLayer.bias.requires_grad = False
                linearLayer.bias.copy_(b.contiguous())
                linearLayer.bias.requires_grad = True

        reorder_head_matrix(self.self.query, index)
        reorder_head_matrix(self.self.key, index)
        reorder_head_matrix(self.self.value, index)
        reorder_head_matrix(self.output.dense, index, dim=1)

    # liux: 加载不同位宽的注意力模块碎片。输入1*12数组，每个数代表一个head碎片的量化位宽。
    # lwg: use similar methods for applying mixed-precision shards 
    # bits: 1x12 array, each entry indicates a bit width for the attention head at its entry's idx
    def patch_attention_shards(self, bits):
        # 12 attention heads, each of 64 dimensions
        n, a = self.self.num_attention_heads, self.self.attention_head_size
        def patch_one_shard(dst, key, bits, dim=0):
            for idx, bit in enumerate(bits):
                if bit not in self.quantized.keys():
                    print("XXXXXXXX requested %d bit not loaded yet!!" % (bit))
                index = torch.arange(idx * a, (idx + 1) * a)
                if key == 'q':
                    src = self.quantized[bit].self.query
                elif key == 'k':
                    src = self.quantized[bit].self.key
                elif key == 'v':
                    src = self.quantized[bit].self.value
                elif key == 'o':
                    src = self.quantized[bit].output.dense
                index = index.to(src.weight.device)
                # liux: clone()返回tensor的拷贝，对象不是同一个并且内存也不是同一块，修改其中一个不会对另外一个造成影响。
                # 但是clone()返回的是中间节点，计算梯度的时候不会保存梯度值，而是直接叠加在原始tensor上，相当于对原始tensor计算梯度。
                # detach()的作用是使得tensor从计算图中脱离出来，使得梯度不会从该tensor传递到下一个tensor。
                # .clone().detach()相当于生成了一个新的tensor，这个新的tensor不仅内存和原tensor不共享，
                # 其后续一系列的tensor操作所造成的梯度，也不会对原tensor造成影响。
                src_w = src.weight.index_select(dim, index).clone().detach() # [768, 64] dim=1, [64, 768] dim=0
                if dst.bias is not None:
                    if dim == 1:
                        src_b = src.bias.clone().detach()
                    else:
                        src_b = src.bias[index].clone().detach()
                    # src_b = src.bias[index].clone().detach()

                # liux: contiguous()函数将tensor的布局变成和行优先，如果不是行优先则新生成一块内存复制数据，如果已经是行优先则什么也不做。
                # tensor在内存中是按照行优先的顺序展开成一维的数组在内存中连续存放(stride()最后一个数是1)，
                # 但是经过transpose等操作后，数据在内存中不动，只是按照下标在内存中寻找数据的方式变了，即stride发生变化(stride()最后一个数不是1)。
                # 对于view()等函数来说，规定其工作方式是在底层内存数据上使用指定的形状查看数据，不能动内存数据，
                # 因此必须要求数据在底层的相邻关系是逻辑上行优先的相邻关系。
                dst.weight.requires_grad = False
                dst.weight.index_copy_(dim, index, src_w.contiguous())
                dst.weight.requires_grad = True

                # liux: TODO 对于attention中dense部分，其碎片是按照dim=1即输入维度划分的，输出维度不切分，因此bias也不应该切分，
                # 但如果每加载一次碎片，就将该碎片完整的bias加载进来，那么最终模型存储的是最后一个碎片的位宽的bias。
                # 因此此处对于dim=1的情况来说，默认不改变bias，使用自己的bias。
                if dst.bias is not None:
                    dst.bias.requires_grad = False
                    if dim==0:
                        dst.bias.index_copy_(0, index, src_b.contiguous())
                    # if dim==1:
                    #     dst.bias.copy_(src_b.contiguous())
                    # else:
                    #     dst.bias.index_copy_(0, index, src_b.contiguous())
                    dst.bias.requires_grad = True

        #print("----before patching---")
        #print(self.self.query.weight)
        patch_one_shard(self.self.query, 'q', bits)
        patch_one_shard(self.self.key, 'k', bits)
        patch_one_shard(self.self.value, 'v', bits)
        patch_one_shard(self.output.dense, 'o', bits, dim=1)
        return


    def add_quantized_model(self, bit, attn):
        self.quantized[bit] = attn

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        # dense layer for adaptive width
        # lwg: hidden size = 768, intermediate_size = 3072
        self.dense = DynaLinear(config.hidden_size, config.intermediate_size,
                                config.num_attention_heads, dyna_dim=[False, True])
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # lwg: XXX
        self.quantized = {}

    def patch_intermediate_shards(self, bits):
        # liux: TODO 这里应该是out_features / num_heads
        a = int(self.dense.out_features / self.dense.num_heads)
        def patch_one_shard(dst, bits, dim=0):
            for idx, bit in enumerate(bits):
                if bit not in self.quantized.keys():
                    print("XXXXXXXX requested %d bit not loaded yet!!" % (bit))
                index = torch.arange(idx * a, (idx + 1) * a)
                src = self.quantized[bit].dense
                index = index.to(src.weight.device)
                src_w = src.weight.index_select(dim, index).clone().detach() #torch.Size([64, 768])???
                if dst.bias is not None:
                    # src_b = src.bias[index].clone().detach()
                    if dim == 1:
                        src_b = src.bias.clone().detach()
                    else:
                        src_b = src.bias[index].clone().detach()
                    # src_b = src.bias[index].clone().detach()

 
                   
                dst.weight.requires_grad = False
                dst.weight.index_copy_(dim, index, src_w.contiguous())
                dst.weight.requires_grad = True

                if dst.bias is not None:
                    dst.bias.requires_grad = False
                    # if dim==1:
                    #     dst.bias.copy_(src_b.contiguous())
                    # else:
                    #     dst.bias.index_copy_(0, index, src_b.contiguous())
                    if dim==0:
                        dst.bias.index_copy_(0, index, src_b.contiguous())
                    dst.bias.requires_grad = True

        patch_one_shard(self.dense, bits)


    def reorder_neurons(self, index, dim=0):
        index = index.to(self.dense.weight.device)
        W = self.dense.weight.index_select(dim, index).clone().detach()
        if self.dense.bias is not None:
            if dim == 1:
                b = self.dense.bias.clone().detach()
            else:
                b = self.dense.bias[index].clone().detach()
        self.dense.weight.requires_grad = False
        self.dense.weight.copy_(W.contiguous())
        self.dense.weight.requires_grad = True
        if self.dense.bias is not None:
            self.dense.bias.requires_grad = False
            self.dense.bias.copy_(b.contiguous())
            self.dense.bias.requires_grad = True

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# lwg: this is actually FFN
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        # dense layer for adaptive width
        self.dense = DynaLinear(config.intermediate_size, config.hidden_size,
                                  config.num_attention_heads, dyna_dim=[True, False])
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # lwg: keep track of quantized siblings 
        self.quantized = {}

    def patch_ffn_shards(self, bits):
        # liux: TODO 这里应该是in_features
        a = int(self.dense.in_features / self.dense.num_heads)
        def patch_one_shard(dst, bits, dim=1):
            for idx, bit in enumerate(bits):
                if bit not in self.quantized.keys():
                    print("XXXXXXXX requested %d bit not loaded yet!!" % (bit))
                index = torch.arange(idx * a, (idx + 1) * a)
                src = self.quantized[bit].dense
                index = index.to(src.weight.device)
                src_w = src.weight.index_select(dim, index).clone().detach()
                if dst.bias is not None:
                    # lwg: we will copy the corresponding bias regardless of dim
                    if dim == 1:
                        src_b = src.bias.clone().detach()
                    else:
                        src_b = src.bias[index].clone().detach()
                    # if dim == 1:
                        # print("dim = 1...", dst.bias.size())
                        # # --- do nothing --- 
                    # else:
                        # #src_b = src.bias[index].clone().detach()
                        # print(dst.bias.size())

                dst.weight.requires_grad = False
                dst.weight.index_copy_(dim, index, src_w.contiguous())
                dst.weight.requires_grad = True

                if dst.bias is not None:
                    dst.bias.requires_grad = False
                    # dst.bias.index_copy_(0, index, src_b.contiguous())
                    # dst.bias.copy_(src_b.contiguous())
                    if dim == 0:
                        dst.bias.index_copy_(0, index, src_b.contiguous())
                    dst.bias.requires_grad = True

        patch_one_shard(self.dense, bits)


    def reorder_neurons(self, index, dim=1):
        index = index.to(self.dense.weight.device)
        W = self.dense.weight.index_select(dim, index).clone().detach()
        if self.dense.bias is not None:
            if dim == 1:
                b = self.dense.bias.clone().detach()
            else:
                b = self.dense.bias[index].clone().detach()
        self.dense.weight.requires_grad = False
        self.dense.weight.copy_(W.contiguous())
        self.dense.weight.requires_grad = True
        if self.dense.bias is not None:
            self.dense.bias.requires_grad = False
            self.dense.bias.copy_(b.contiguous())
            self.dense.bias.requires_grad = True

    # liux: hidden_states: intermediate_output, input_tensor: attention_output
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.output_intermediate = config.output_intermediate

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        if self.output_intermediate:
            outputs = (layer_output,) + attention_outputs[1:] + (intermediate_output,)
        else:
            outputs = (layer_output,) + attention_outputs[1:]

        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        # liux: 下面三个变量都是bool变量。
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.output_intermediate = config.output_intermediate
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        # liux: 模型深度比值。
        self.depth_mult = 1.
        # lwg: orig precsion bit = 0
        self.bit = 0
        # point to other quantized models
        self.quantized = {}

    def add_quantized_model(self, bert_encoder):
        bit = bert_encoder.bit
        for i in range(len(self.layer)):
            print("adding %d layer in %d bits" % (i, bit))
            self.layer[i].attention.quantized[bit] = bert_encoder.layer[i].attention
            self.layer[i].intermediate.quantized[bit] = bert_encoder.layer[i].intermediate
            self.layer[i].output.quantized[bit] = bert_encoder.layer[i].output
        return

    def update_bit(self, bit):
        self.bit = bit

    def quantize(self, bits):
        #for i in range(6, 12):
        #for i in np.random.choice(12, 6, replace=False):
        outliers = 0
        for i in range(len(self.layer)):
            #gobo_quantize_one_layer(self.layer[i], 6)
            #kmeans_quantize_one_layer(self.layer[i], 3)
            print("------------------quantize Encoder %d-th Layer------------------" % i)
            outliers += _quantize(self.layer[i], gobo_quantize, True, bits)
            print("outliers = %d" % outliers)

        print("------------------  outliers: {:.3%}".format(outliers/(7087872*12.0)))
        #print("------------------  outliers: %d" % outliers)


    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        #start = time.time()
        all_hidden_states = ()
        all_attentions = ()
        all_intermediate = ()

        # uniformly remove layers
        depth = round(self.depth_mult * len(self.layer))
        kept_layers_index = []

        # (0,2,4,6,8,10)
        for i in range(depth):
            kept_layers_index.append(math.floor(i/self.depth_mult))

        # print("keeping...", kept_layers_index)

        for i in kept_layers_index:
            layer_module = self.layer[i]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if self.output_intermediate:
                all_intermediate = all_intermediate + (layer_outputs[2],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        if self.output_intermediate:
            outputs = outputs + (all_intermediate,)
        #end = time.time()
        #print(self, "forwarding in Encoder time: ", (end-start)*1000)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        # lwg: 768 x 768
        # lwg: shall we take a slice as well??
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617

            # lwg: zero out...
            #module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.zero_()
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        # liux: 如果最外层调用forward传入的attention_mask为None，则不会mask任何值。
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            Attentions scores before softmax.
        **intermediates**: (`optional`, returned when ``config.output_intermediate=True``)
            representation in the intermediate layer after nonlinearity.
    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # lwg: 768 x 2
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  regression task
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions), (intermediates)


