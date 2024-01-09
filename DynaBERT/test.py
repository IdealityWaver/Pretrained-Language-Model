from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm

from transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                          )

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from sti_plan import plan
import math



logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='GLUE\RTE', type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_name", default='RTE', type=str, required=False,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default='output', type=str, required=False,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--model_dir", type=str, default='D:\\123\\Code\\DynaBertModels',
                        help="The teacher model dir.")
    parser.add_argument('--depth_mult', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")
    parser.add_argument('--width_mult', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")
    parser.add_argument('--emb', type=int, default=32,
                        help="Embedding quantization bits")
    parser.add_argument('--enc', type=int, default=32,
                        help="Encoder quantization bits")
    args = parser.parse_args()



    # liux: 准备对应任务的模型目录和device。
    args.model_dir = os.path.join(args.model_dir, args.task_name)
    model_root = args.model_dir
    bits_conf = str(args.emb) + '_' + str(args.enc)
    args.model_dir = os.path.join(model_root, bits_conf)
    device = torch.device("cuda:3" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    torch.cuda.empty_cache()
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu, )
    set_seed(args)


    # liux: 准备GLUE任务数据集。
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)


    # liux: 读取对应量化位目录下的模型config、模型和tokenizer，并加载archive文件中的模型参数到模型实例中。
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_dir, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_dir, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_dir, config=config)
    model.to(args.device)
    model.bert.encoder.update_bit(args.enc)


    # liux: 量化模型至不同的位宽，并保存模型。
    '''
    emb_bits = 3
    enc_bits = 4
    if emb_bits != 0:
        print("--------quantize embedings--------")
        model.bert.embeddings.quantize(emb_bits)
    if enc_bits != 0:
        print("--------quantize encoder--------")
        model.bert.encoder.quantize(enc_bits)
        model.bert.encoder.update_bit(enc_bits)
    model_save_dir = os.path.join(model_root, str(emb_bits) + '_' + str(enc_bits))
    if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
    model.save_pretrained(model_save_dir)
    tokenizer.save_vocabulary(model_save_dir)
    '''

    # liux: 从文件中加载特定量化位的模型，并将该量化后的enc添加到主模型的数组中保存。
    def load_quantized_model(enc_bit, emb_bit=3):
        bits_conf = str(emb_bit) + '_' + str(enc_bit)
        print("loading ", bits_conf, "...")
        model_dir = os.path.join(model_root, bits_conf)
        config = config_class.from_pretrained(model_dir, num_labels=num_labels, finetuning_task=args.task_name)
        tokenizer = tokenizer_class.from_pretrained(model_dir, do_lower_case=args.do_lower_case)
        _model = model_class.from_pretrained(model_dir, config=config)
        _model.to(args.device)
        _model.bert.encoder.update_bit(enc_bit)
        model.bert.encoder.add_quantized_model(_model.bert.encoder)
        # 64 dim per head
        #print(model.bert.encoder.layer[0].attention.self.attention_head_size)
        return _model

    # load_quantized_model(4)
    # load_quantized_model(6)
    # load_quantized_model(8)
    # load_quantized_model(10)

    # shard_conf1 = [4] * 12
    #shard_conf1[3] = 6 
    #shard_conf1[0] = 6 
    # model.bert.encoder.layer[0].attention.patch_attention_shards(shard_conf1)
    # model.bert.encoder.layer[0].intermediate.patch_intermediate_shards(shard_conf1)
    # model.bert.encoder.layer[0].output.patch_ffn_shards(shard_conf1)
    # for i in range(12):
    #     model.bert.encoder.layer[i].attention.patch_attention_shards(shard_conf1)
    #     model.bert.encoder.layer[i].intermediate.patch_intermediate_shards(shard_conf1)
    #     model.bert.encoder.layer[i].output.patch_ffn_shards(shard_conf1)

    def patch_layer_shard(l, conf):
        print("Patch Layer {0} to {1}".format(l, conf))
        model.bert.encoder.layer[l].attention.patch_attention_shards(conf)
        model.bert.encoder.layer[l].intermediate.patch_intermediate_shards(conf)
        model.bert.encoder.layer[l].output.patch_ffn_shards(conf)

    def write_to_results(s):
        eval_output_dir = os.path.join(args.output_dir,
                                       args.model_type + '_' + args.width_mult + '_' + args.depth_mult + '_eval')
        if not os.path.exists(eval_output_dir):
                # and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        #output_eval_file = os.path.join(eval_output_dir, "{0}_prof.txt".format("ablation_upgrade"))
        output_eval_file = os.path.join(eval_output_dir, "{0}_ours.txt".format(args.task_name))
        with open(output_eval_file, "a") as writer:
            writer.write(s)
            writer.write('\n')

    from datetime import datetime
    exp_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    write_to_results("%s eval begin at %s" % args.task_name, exp_time)


    


if __name__ == "__main__":
    # a=np.array([1,2,3])
    # mask = np.zeros(torch.Size([3]), dtype=bool)
    # mask[np.where(a >= 2.0)] = True
    # print(mask)
    # print(a[mask])
    # print(~mask)
    # print(a[~mask])

    # x = torch.tensor([2, 10, 5])
    # mask = torch.tensor([True, False, True],dtype=torch.bool)
    # maskk = np.array([True, False, False])
    # print(x[maskk])
    # print(x[~mask])
    main()


    