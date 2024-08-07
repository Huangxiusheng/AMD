import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'

import torch



from easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons, model_type
from transformers import  LlamaForCausalLM,LlamaTokenizer, AutoModelForCausalLM, AutoConfig
import json
import itertools
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

import transformers
from quant import QuantLinear
import torch.nn as nn
from accelerate import dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
import accelerate
def distribute_model(model_adapter) -> None:
    """Distribute the model across available GPUs."""
    # model = model_adapter.model
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



def make_quant(module, names, bits, groupsize, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_quant(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

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


# def load_quant(model, checkpoint, wbits, groupsize, device_map):
#     from transformers import LlamaConfig, LlamaForCausalLM 
#     config = LlamaConfig.from_pretrained(model)
#     def noop(*args, **kwargs):
#         pass
#     torch.nn.init.kaiming_uniform_ = noop 
#     torch.nn.init.uniform_ = noop 
#     torch.nn.init.normal_ = noop 

#     torch.set_default_dtype(torch.half)
#     transformers.modeling_utils._init_weights = False
#     torch.set_default_dtype(torch.half)
#     with accelerate.init_empty_weights():
#         model = LlamaForCausalLM(config)
#         torch.set_default_dtype(torch.float)
#         model = model.eval()
#         layers = find_layers(model)
#         for name in ['lm_head']:
#             if name in layers:
#                 del layers[name]
#         make_quant(model, layers, wbits, groupsize)


#     print('Loading model ...')
#     model = accelerate.load_checkpoint_and_dispatch(model, checkpoint, device_map=device_map, no_split_module_classes=['LlamaDecoderLayer'])
#     model.seqlen = 2048
#     print('Done.')

#     return model



with open('test.json','r') as f:
    knowns = json.load(f)
def hausdorff_distance(set_a, set_b):
    return max(directed_hausdorff(set_a, set_b)[0], directed_hausdorff(set_b, set_a)[0])

# 定义 Jaccard 相似系数函数
def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union

model_name = "/share/projset/hxs-6k/huangxiusheng/Model_edit/model_saves/models--meta-llama--Llama-2-7b-hf"
# model_name = "/share/projset/hxs-6k/huangxiusheng/Model_edit/memit/EleutherAI/gpt-j-6b"
torch_dtype = torch.float16
quant = True
if not quant:
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map='auto')
else:

    checkpoint1 = '/share/projset/hxs-6k/huangxiusheng/AMD/GPTQ-for-LLaMa/llama7b-4bit-128g.pt'
    wbits = 4
    groupsize = 128
    # device1 = 'auto'
    device1 = -1
    model = load_quant(model_name, checkpoint1, wbits, groupsize, device1)
    model.to('cuda')
    # distribute_model(model)




tok = LlamaTokenizer.from_pretrained(model_name)
tok.pad_token_id = tok.eos_token_id

kn = KnowledgeNeurons(
    model,
    tok,
    model_type=model_type(model_name),
    device=f"cuda:0",
)
zong = 0
single = 0
for id, known in tqdm(enumerate(knowns)):
    sets = []
    for prompt in known:
        
        text = [prompt['question'],]
        ground_truth = prompt['target']
        refined_neurons = kn.get_refined_neurons(
            text,
            ground_truth,
            p=0.4,
            batch_size=1,
            steps=1,
            coarse_adaptive_threshold=0.2,
            refine=False,
        )
        refined_neurons_copy = []
        for a in refined_neurons:
            refined_neurons_copy.append(a)
        sets.append(refined_neurons_copy)
    # 生成所有集合对的组合
    len_all = []
    for set_single in sets:
        len_all.append(len(set_single))
    zong += max(len_all)
    combinations = itertools.combinations(range(len(sets)), 2)
    flag = []
    for (i, j) in combinations:
        # 将每个子列表转换为元组
        set_a = set(tuple(x) for x in sets[i])
        set_b = set(tuple(x) for x in sets[j])

        # 求交集
        intersection_set = set_a.intersection(set_b)

        # 将结果转换回列表形式
        intersection_list = [list(x) for x in intersection_set]
        flag.append(len(intersection_list))
    single += max(flag)
    print('####################:  ' + str(id))
    print('id: {}, all_number: {}'.format(id,zong))
    print('id: {}, equal_number: {}'.format(id,single))
    if id % 1000 == 0:
        print('id: {}, all_number: {}'.format(id,zong))
        print('id: {}, equal_number: {}'.format(id,single))
        if id > 0:
            break
a = 0
