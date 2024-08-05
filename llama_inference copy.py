import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *
from ppl import evaluate_ppl
from transformers import AutoTokenizer
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

def distribute_model(model_adapter) -> None:
    """Distribute the model across available GPUs."""
    model = model_adapter.model
    max_memory = get_balanced_memory(
        model,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory,
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )


DEV = torch.device('cuda')

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

def load_quant(model, checkpoint, wbits, groupsize, device):
    from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    config = AutoConfig.from_pretrained(model, trust_remote_code=True,)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,torch_dtype=torch.float16)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        if device == -1:
            device = "cpu"
        model.load_state_dict(safe_load(checkpoint, device))
    else:
        model.load_state_dict(torch.load(checkpoint))
        # model.tie_weights()
        # model = load_checkpoint_and_dispatch(model, checkpoint, device_map="auto")

    model.seqlen = 2048
    print('Done.')

    return model

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()
    # 
    parser.add_argument(
        '--model', type=str,
        # default='/share/project/lixiang/flm_models/7b/chat/iter_0001534_HF_FLM2_llamafy',
        # default='/share/project/lixiang/flm_models/sft/sft_52b_364000_math_p50_code_p25_dxdl_p25_yq_best_bs128_nwd0.01_c2_lr0.5/iter_0001440_HF_FLM2_llamafy',
        default='/share/projset/hxs-6k/huangxiusheng/Model_edit/EasyEdit-main/model_saves/models--meta-llama--Llama-2-7b-hf',
        help='llama model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=4, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=128,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--load', type=str, default='llama7b-4bit-128g.pt',
        help='Load quantized model.'
    )

    parser.add_argument(
        '--text', type=str,default="this is llama",
        help='input text'
    )
    
    parser.add_argument(
        '--min_length', type=int, default=10,
        help='The minimum length of the sequence to be generated.'
    )
    
    parser.add_argument(
        '--max_length', type=int, default=50,
        help='The maximum length of the sequence to be generated.'
    )
    
    parser.add_argument(
        '--top_p', type=float , default=0.95,
        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )
    
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The value used to module the next token probabilities.'
    )

    parser.add_argument(
        '--device', type=int, default=-1,
        help='The device used to load the model when using safetensors. Default device is "cpu" or specify, 0,1,2,3,... for GPU device.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],default='wikitext2',
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )



    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()
    # model = torch.load('/share/projset/hxs-6k/huangxiusheng/AMD/GPTQ-for-LLaMa/new_4bit.pt')
    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize, args.device)
    else:
        model = get_llama(args.model)
        model.eval()
    model.to('cuda:0')
    # distribute_model(model)
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())

    # 计算模型大小（假设每个参数为4字节）
    model_size_gb = total_params * 4 / (1024 ** 3)

    print(f"Total parameters: {total_params}")
    print(f"Model size (in GB): {model_size_gb:.2f} GB")
    # model = model.model
    dataset_ppl = evaluate_ppl(model, model.config.pad_token_id, testloader)
    a = 0






    
    # print(dataset_ppl)

    # model.to(DEV)
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    # input_ids = tokenizer.encode(args.text, return_tensors="pt").to(DEV)

    # with torch.no_grad():
    #     generated_ids = model.generate(
    #         input_ids,
    #         do_sample=True,
    #         min_length=args.min_length,
    #         max_length=args.max_length,
    #         top_p=args.top_p,
    #         temperature=args.temperature,
    #     )
    # print(tokenizer.decode([el.item() for el in generated_ids[0]]))
