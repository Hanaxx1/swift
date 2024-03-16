# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset as HfDataset
from modelscope import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                        MsDataset)
from torch import dtype as Dtype
from transformers.utils.versions import require_version

from swift.llm import (LoRATM, Template, TemplateType, dataset_map,
                       get_dataset, get_model_tokenizer, get_template,
                       print_example, register_dataset, register_model,
                       register_template)
from swift.utils import get_logger

logger = get_logger()

class CustomModelType:
    tigerbot_7b = 'tigerbot-7b'
    tigerbot_13b = 'tigerbot-13b'
    tigerbot_13b_chat = 'tigerbot-13b-chat'

    orca2_7b = "orca-2-13b"
    openchat_35 = "openchat_3.5"
    fusechat_7b = "fusechat-7b-varm"
    mistral_7b_instruct = "mistral-7b-instruct-v0.2"
    neural_chat_7b = "neural-chat-7b-v3"
    solar_instruct_10_7b = "solar-10.7b-instruct"
    solar_instruct_10_7b_128k = "solar-10.7b-instruct-128k"
    sauerkrautlm_solar_instruct = "sauerkrautlm-solar-instruct"
    marcoroni_7b = "marcoroni-7b-v3"
    dpopenHermes_7b = "dpopenHermes-7b-v2"
    neuralbeagle14 = "neuralbeagle14-7B"
    turdus = "turdus"
    darebeagle_7b = "darebeagle-7b-v2"
    una_thebeagle_7b = "una-thebeagle-7b-v1"
    mixtral_moe_7b_instruct_gptq_int4 = "mixtral-8x7B-instruct-v0.1-gptq-int4"
    gemma_7b_it = 'gemma-7b-it'
    merlinite_7b = 'merlinite-7b'

class CustomTemplateType:
    tigerbot = 'tigerbot'

    orca2 = "orca-2"
    openchat_35 = "openchat_3.5"
    neural = "neural"
    solar = "solar"
    marcoroni = "marcoroni"
    mistral = "mistral"
    chatml = "_chatml" # 无system message的chatml
    llama = "_llama" # 无system message的llama
    merlinite = "merlinite"

class CustomDatasetName:
    stsb_en = 'stsb-en'


@register_model(CustomModelType.tigerbot_7b,
                'TigerResearch/tigerbot-7b-base-v3', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(CustomModelType.tigerbot_13b,
                'TigerResearch/tigerbot-13b-base-v2', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(CustomModelType.tigerbot_13b_chat,
                'TigerResearch/tigerbot-13b-chat-v4', LoRATM.llama2,
                CustomTemplateType.tigerbot)
def get_tigerbot_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if use_flash_attn:
        require_version('transformers>=4.34')
        logger.info('Setting use_flash_attention_2: True')
        model_kwargs['use_flash_attention_2'] = True
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs)
    return model, tokenizer


@register_model(CustomModelType.orca2_7b,
                '/home/css/models/Orca-2-13b', LoRATM.llama2,
                CustomTemplateType.orca2)
def get_orca2_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir)
    # model_config.pretraining_tp = 1
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            **model_kwargs)
    return model, tokenizer

@register_model(CustomModelType.openchat_35,
                '/home/css/models/openchat-3.5-0106', LoRATM.llama2,
                CustomTemplateType.openchat_35)
@register_model(CustomModelType.fusechat_7b,
                '/home/css/models/FuseChat-7B-VaRM', LoRATM.llama2,
                CustomTemplateType.openchat_35)
@register_model(CustomModelType.mistral_7b_instruct,
                '/home/css/models/Mistral-7B-Instruct-v0.2', LoRATM.llama2,
                CustomTemplateType.mistral)
@register_model(CustomModelType.neural_chat_7b,
                '/home/css/models/neural-chat-7b-v3-3', LoRATM.llama2,
                CustomTemplateType.neural)
@register_model(CustomModelType.solar_instruct_10_7b,
                '/home/css/models/SOLAR-10.7B-Instruct-v1.0', LoRATM.llama2,
                CustomTemplateType.solar)
@register_model(CustomModelType.sauerkrautlm_solar_instruct,
                '/home/css/models/SauerkrautLM-SOLAR-Instruct', LoRATM.llama2,
                CustomTemplateType.solar)
@register_model(CustomModelType.marcoroni_7b,
                '/home/css/models/Marcoroni-7B-v3', LoRATM.llama2,
                CustomTemplateType.marcoroni)
@register_model(CustomModelType.dpopenHermes_7b,
                '/home/css/models/DPOpenHermes-7B-v2', LoRATM.llama2,
                CustomTemplateType.chatml)
@register_model(CustomModelType.una_thebeagle_7b,
                '/home/css/models/UNA-TheBeagle-7b-v1', LoRATM.llama2,
                CustomTemplateType.neural)
@register_model(CustomModelType.turdus,
                '/home/css/models/Turdus', LoRATM.llama2,
                CustomTemplateType.llama # neural 也可以
)
@register_model(CustomModelType.darebeagle_7b,
                '/home/css/models/DareBeagle-7B-v2', LoRATM.llama2,
                CustomTemplateType.neural)
@register_model(CustomModelType.neuralbeagle14,
                '/home/css/models/NeuralBeagle14-7B', LoRATM.llama2,
                CustomTemplateType.neural)
@register_model(CustomModelType.mixtral_moe_7b_instruct_gptq_int4,
                '/home/css/models/Mixtral-8x7B-Instruct-v0.1-GPTQ-int4', LoRATM.llama2,
                CustomTemplateType.mistral)
@register_model(CustomModelType.gemma_7b_it,
                '/home/css/models/gemma-7b-it', LoRATM.llama2,
                TemplateType.gemma)
@register_model(CustomModelType.merlinite_7b,
                '/home/css/models/merlinite-7b', LoRATM.llama2,
                CustomTemplateType.merlinite)
def get_model_tokenizer(
    model_dir: str,
    torch_dtype: Dtype, 
    model_kwargs: Dict[str, Any], 
    load_model: bool = True,
    **kwargs
):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.torch_dtype = torch_dtype
    model_config._attn_implementation = 'eager'
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            **model_kwargs)
    return model, tokenizer

@register_model(CustomModelType.solar_instruct_10_7b_128k,
                '/home/css/models/SOLAR-10B-Instruct-v1-128k', LoRATM.llama2,
                CustomTemplateType.solar)
def get_yarn_model_tokenizer(
    model_dir: str,
    torch_dtype: Dtype, 
    model_kwargs: Dict[str, Any], 
    load_model: bool = True,
    **kwargs
):
    # model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # model_config.torch_dtype = torch_dtype
    # logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = None
    
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            attn_implementation="flash_attention_2",
            # config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs)
    return model, tokenizer

# Ref: https://github.com/TigerResearch/TigerBot/blob/main/infer.py
register_template(
    CustomTemplateType.tigerbot,
    Template(['{{SYSTEM}}'],
             ['\n\n### Instruction:\n{{QUERY}}\n\n### Response:\n'], [],
             [['eos_token_id']]))

register_template(
    CustomTemplateType.openchat_35,
    Template(
        [],
        ['GPT4 Correct User: {{QUERY}}<|end_of_turn|>GPT4 Correct Assistant:'],
        ['<|end_of_turn|>'], ['<|end_of_turn|>'], None, ['{{SYSTEM}}<|end_of_turn|>']))

# 不支持多轮对话
register_template(
    CustomTemplateType.orca2,
    Template(
        [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>'], None,
        ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))

register_template(
    CustomTemplateType.neural,
    Template( 
        [],
        ['### User:\n{{QUERY}}\n### Assistant:\n'],
        ['\n'], ['</s>'], None, ['### System:\n{{SYSTEM}}\n']))

register_template(
    CustomTemplateType.solar,
    Template(
        [],
        ['### User:\n{{QUERY}}\n\n### Assistant:\n'],
        ['\n\n'], ['</s>'], None, ['### System:\n{{SYSTEM}}\n\n']))

register_template(
    CustomTemplateType.marcoroni,
    Template(
        [],
        ['### Instruction:\n\n{{QUERY}}\n\n### Response:\n'],
        ['\n\n'], ['</s>'], None, ['### System:\n\n{{SYSTEM}}\n\n']))

register_template(
    CustomTemplateType.mistral,
    Template(
        ['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '], 
        ['</s>'], None,
        ['<s>[INST] {{SYSTEM}}\n']))

register_template(
    CustomTemplateType.chatml,
    Template(
        [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>'], None,
        ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))

# default system: "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
register_template(
    CustomTemplateType.merlinite,
    Template(
        [], ['<|user|>\n{{QUERY}}\n<|assistant|>\n'],
        ['<|endoftext|>\n'], ['<|endoftext|>'], None,
        ['<|system|>\n{{SYSTEM}}\n']))

# 无限回答的问题
# register_template(
#     CustomTemplateType.neuralbeagle14,
#     # Template(
#     #     ['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '], 
#     #     ['</s>'], None,
#     #     ['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n'])
#     Template(
#         [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
#         ['<|im_end|>\n'], ['<|im_end|>'], None,
#         ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
# )

register_template(
    CustomTemplateType.llama,
    Template(
        ['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '],
        ['</s>'], None,
        ['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n'])
)

def _preprocess_stsb(dataset: HfDataset) -> HfDataset:
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """
    query = []
    response = []
    for d in dataset:
        query.append(prompt.format(text1=d['text1'], text2=d['text2']))
        response.append(f"{d['label']:.1f}")
    return HfDataset.from_dict({'query': query, 'response': response})


@register_dataset(
    CustomDatasetName.stsb_en, 'huangjintao/stsb', task='text-generation')
def get_stsb_dataset(dataset_id_or_path: str,
                     **kwargs) -> Tuple[HfDataset, Optional[HfDataset]]:
    dataset_dict = MsDataset.load(dataset_id_or_path)
    train_dataset = dataset_dict['train'].to_hf_dataset()
    val_dataset = dataset_dict['validation'].to_hf_dataset()
    return tuple(
        _preprocess_stsb(dataset) for dataset in [train_dataset, val_dataset])


if __name__ == '__main__':
    # The Shell script can view `examples/pytorch/llm/scripts/custom`.
    # test dataset
    train_dataset, val_dataset = get_dataset([CustomDatasetName.stsb_en],
                                             check_dataset_strategy='warning')
    print(f'train_dataset: {train_dataset}')
    print(f'val_dataset: {val_dataset}')
    # test model base
    model, tokenizer = get_model_tokenizer(
        CustomModelType.tigerbot_13b, use_flash_attn=False)
    # test model chat
    model, tokenizer = get_model_tokenizer(
        CustomModelType.tigerbot_13b_chat, use_flash_attn=False)
    # test template
    template = get_template(CustomTemplateType.tigerbot, tokenizer)
    train_dataset = dataset_map(train_dataset, template.encode)
    print_example(train_dataset[0], tokenizer)