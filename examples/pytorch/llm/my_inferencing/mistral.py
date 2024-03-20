import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm, VllmGenerationConfig
)
from swift.utils import seed_everything
from swift.tuners import Swift
import json

#VllmConfig
generation_config = VllmGenerationConfig(
    max_new_tokens=4096,
    temperature=0,
)


import sys
print(sys.path)
sys.path.append('/home/qiuyang/workplace/swift/examples/pytorch/llm/')

import custom

#微调目录
model_dir1 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/mistral-7b-chat-v2/v7-20231228-133749/checkpoint-17477'
model_dir2 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/mistral-7b-chat-v2/v9-20240104-151231/checkpoint-250'
model_dir5 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/mistral-7b-chat-v2/v10-20240120-151243/checkpoint-1948'

#llama
model_dir3 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/llama2-7b-chat/v4-20240113-171722/checkpoint-17477'
model_dir4 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/llama2-7b-chat/v5-20240115-130325/checkpoint-250'
model_dir6 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/llama2-7b-chat/v22-20240129-233014/checkpoint-1945'

#openchat
model_dir7 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/openchat_3.5/v0-20240131-124616/checkpoint-1948'

#solar
model_dir8 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/solar-10.7b-instruct/v1-20240218-003119/checkpoint-1948'

# mistral
model_type1 = ModelType.mistral_7b_chat_v2

mistral_engine = get_vllm_engine(
    model_type1, 
    model_id_or_path = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/mistral-7b-chat-v2/v10-20240120-151243/checkpoint-1948-merged',
    tensor_parallel_size=2,
    seed=42
)

template_type = get_default_template_type(model_type1)
template = get_template(template_type, mistral_engine.hf_tokenizer)


template_type1 = get_default_template_type(model_type1)
model_mistral, tokenizer_mistral = get_model_tokenizer(model_type1, model_kwargs={'device_map': 'auto'}, model_dir="/home/css/models/Mistral-7B-Instruct-v0.2")
model_mistral.generation_config.max_new_tokens = 4096
# model_mistral = Swift.from_pretrained(model_mistral, model_dir1, inference_mode=True)
# model_mistral = Swift.from_pretrained(model_mistral, model_dir5, inference_mode=True)
template_mistral = get_template(template_type1, tokenizer_mistral)

#zephyr
# model_type2 = ModelType.zephyr_7b_beta_chat
# template_type2 = get_default_template_type(model_type2)
# model_zephyr, tokenizer_zephyr = get_model_tokenizer(model_type2, model_kwargs={'device_map': 'auto'}, model_dir="/home/css/models/zephyr-7b-beta")
# model_zephyr.generation_config.max_new_tokens = 4096
# model_zephyr = Swift.from_pretrained(model_zephyr, model_dir5, inference_mode=True)
# template_zephyr = get_template(template_type2, tokenizer_zephyr)


#llama
# model_type3 = ModelType.llama2_7b_chat
# template_type3 = get_default_template_type(model_type3)
# model_llama, tokenizer_llama = get_model_tokenizer(model_type3, model_kwargs={'device_map': 'auto'}, model_dir="/home/css/models/Llama-2-7b-chat-hf")
# model_llama.generation_config.max_new_tokens = 4096
# # model_llama = Swift.from_pretrained(model_llama, model_dir3, inference_mode=True)
# model_llama = Swift.from_pretrained(model_llama, model_dir6, inference_mode=True)
# template_llama = get_template(template_type3, tokenizer_llama)


#openchat
model_type4 = custom.CustomModelType.openchat_35
template_type4 = custom.CustomTemplateType.openchat_35
model_openchat, tokenizer_openchat = get_model_tokenizer(model_type4, model_kwargs={'device_map': 'auto'}, model_dir="/home/css/models/openchat-3.5-0106",load_model=False)
#,load_model=False
# model_openchat.generation_config.max_new_tokens = 4096
# # model_llama = Swift.from_pretrained(model_llama, model_dir3, inference_mode=True)
# model_openchat = Swift.from_pretrained(model_openchat, model_dir7, inference_mode=True)
# template_openchat = get_template(template_type4, tokenizer_openchat)

#solar
# model_type5 = custom.CustomModelType.solar_instruct_10_7b
# template_type5 = get_default_template_type(model_type5)
# model_solar, tokenizer_solar = get_model_tokenizer(model_type5, model_kwargs={'device_map': 'auto'}, model_dir="/home/css/models/SOLAR-10.7B-Instruct-v1.0")
# model_solar.generation_config.max_new_tokens = 4096
# model_solar = Swift.from_pretrained(model_solar, model_dir8, inference_mode=True)
# template_solar = get_template(template_type5, tokenizer_solar)


# model_llama.generation_config.top_p=0.75
# model_llama.generation_config.temperature=0.01
# model_llama.generation_config.update(repetition_penalty=2.0,top_k=40,num_beams=1)


# model_mistral.generation_config.top_p=0.75
# model_mistral.generation_config.temperature=0.01
# model_mistral.generation_config.update(repetition_penalty=2.0,top_k=40,num_beams=2)
# , do_sample=True
# content = '''SaharaReporters has learned from reliable sources at the Nigerian Presidency that President Muhammadu Buhari, who has been in the UK since January 19, is receiving \u201cintense treatment\u201d for a renewed flare-up of prostate issues. He had undergone surgical treatment for prostate cancer soon after losing the 2011 presidential election to former President Goodluck Jonathan.\nIn addition to his UK doctors, President Buhari has been under the care of two Nigerian physicians, Dr. Suhayb Sanusi Rafindadi, who is his Chief Personal Physician, and Dr. Ugorji Ogbonna, who had a medical practice for many years in Kano before relocating to the UK. Even though he is from the southeast of Nigeria, Dr. Ogbonna is very close to numerous powerful northern politicians, and one of his sons converted to Islam, according to a source. While Dr. Rafindadi accompanied the president on his medical trips abroad and Dr. Ogbonna often liaise with Mr. Buhari\u2019s physicians in the UK to arrange for his treatments.\nOur sources indicated that Mr. Buhari\u2019s original surgical treatment for prostate cancer had been declared successful. However, when a nagging ear infection forced President Buhari to visit his longtime physician in the UK last year, the prognosis led the president to seek two separate medical opinions, in France and later Germany. Doctors in both countries reportedly told him it was urgent to see his doctors in the UK. Subsequently, Mr. Buhari, who was visiting France for a conference, moved from Paris to London, declaring that he needed a vacation.\nMr. Buhari\u2019s UK doctors advised that he needed to stay put in London for another surgery, but the president bowed to pressure from members of his inner circle and decided to make a premature return to Abuja out of political expediency. However, before President Buhari left London, his doctors there removed polyps from his nostrils in a surgical procedure to ease his breathing.\nIn 2016, President Buhari made trips to the UK in February and June to consult his UK doctors on a variety of health issues, including his ear infection, said our sources.\nSaharaReporters learned that President Buhari currently receives intense treatment for a prostate-related ailment. The treatment at a point severely affected his voice and appetite. He has progressively lost weight and has had to be force-fed on occasion on the orders of his doctors. One source said Mr. Buhari\u2019s treatments had been compounded by his age, which his UK doctors believe to be more than 80 rather than his official \u201cage\u201d of 74 years old.\nInformation gleaned from our sources indicated an attempt to obfuscate the precise nature of Mr. Buhari\u2019s illness, signaling efforts by competing factions around the president to keep the Nigerian public uninformed. According to our sources, four different versions of President Buhari's condition were being circulated. One version, traced to his wife, Aisha Buhari, is that the president has \u201cinternal organ\u201d issues.\nOur sources disclosed that a cabal led by President Buhari's cousin, Mamman Daura, reportedly sent Mrs. Aisha Buhari away from the UK, asking her to make a temporary visit to Saudi Arabia. To justify Aisha Buhari\u2019s exit from London, Mr. Daura reportedly claimed that the president\u2019s condition appeared to worsen each time his wife was around him. He reportedly told Mrs. Buhari that part of her husband's illness was a result of \u201ca spiritual attack,\u201d and asked her to proceed to Saudi Arabia to pray for him. From Saudi Arabia, Mrs. Buhari returned to Nigeria about two weeks ago. One source claimed that she had not returned to London ever since.\nOur sources said Mr. Daura often handled the daily briefing of select Presidency officials about the president's health. According to them, the president\u2019s cousin continues to claim that Mr. Buhari was only exhausted and needed adequate rest before returning to Nigeria ready to take over the mantle of leadership once again.\nAs SaharaReporters revealed last weekend, the doctors treating Mr. Buhari in the UK have told him in clear terms that he ought to shelf any plans to return to serious work and stay back in London for as long as four months to receive a full course of treatment. One source indicated that members of the president\u2019s inner circle were yet to fully embrace the doctors\u2019 recommendation and communicate the information to Nigerians because they fear the loss of political influence.\nOn his part, President Buhari had reportedly told his inner political circle that he was in no hurry to return to Nigeria, adding that he was willing to let the expert advice of his doctors to prevail. However, some of his die-hard associates have not given up plotting to daily to convince him to return to Abuja in defiance of medical advice.\nPresident Buhari reportedly communicates regularly with Acting President Yemi Osinbajo, encouraging him to carry on the task of governing Nigeria while he undertakes necessary treatment needed to keep him alive.'''
content = '''"Ukranian Prime Min Viktor F Yanukovich visits Washington for low key meetings with Vice Pres Dick Cheney and Sec of State Condoleezza Rice; says Ukraine is in no hurry to join NATO or European Union (S)"'''
eval_prompt = f"### Instruction: According to the following information:<{content}> Answer the question. Whether anyone in the information was transferred between countries in text?if so,from which country to which country?What year and month did the transfer take place?only give the starting point and destination example [Yes or No,starting point,destination,year-month],just country name.\n### Response:\n"

# print(eval_prompt)

# response_llama, history_llama = inference(
#     model_mistral, template_mistral, eval_prompt,
# )

# print(response_llama)

content_list = []
# count = 0
# 打开JSON文件
with open('/home/qiuyang/workplace/swift/examples/pytorch/llm/my_tuning/data/c4_data7.json', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 解析JSON数据
        data = json.loads(line)

        if len(tokenizer_openchat.tokenize(data['content'])) <3500:
            # count = count + 1
            print(len(tokenizer_openchat.tokenize(data['content'])))
        # 在这里可以对每一行的数据进行处理
            content_list.append(data['content'])


print(len(content_list))
result_mistral = []

for i in range(0,len(content_list)):
    print(i)
    content = content_list[i]
    eval_prompt = f"### Instruction: According to the following information:<{content}> Answer the question。Whether anyone in the information was transferred between countries in text?if so,from which country to which country?What year and month did the transfer take place?only give the starting point and destination example [Yes or No,starting point,destination,year-month],just country name.\n### Response:\n"
    # response_mistral, history_mistral = inference(model_llama, template_llama, eval_prompt)
    # response_zephyr, history_zephyr = inference(model_zephyr, template_zephyr, eval_prompt)

    request_list = [
    # {'query': "Below is a CLAIM and some INFORMATION searched online. These pieces of INFORMATION are relevant to the CLAIM. This CLAIM and all INFORMATION include their respective publication dates and contents. To classify the CLAIM more accurately (if the content described by the CLAIM is correct, it will be classified as TRUE; if the content described by the CLAIM is incorrect, it will be classified as FALSE), please first expand on the given INFORMATION and provide a detailed summary of it. Then analyze, reason, and provide reasonable evidence to judge the correctness of the CLAIM based on the available information and your knowledge, and finally generate prior knowledge that helps classify the CLAIM.\n\nCLAIM:\nPublication date: 2020-08-13\nContent: Illinois Strengthens Face Mask Rules in Businesses\n\nINFORMATION:\nInformation 1:\nPublication date: None\nTitle: FAQ for Businesses Concerning Use of Face-Coverings During COVID-19\nContent:\nA: Businesses reserve the right to refuse service to persons unable to comply with the requirement to wear a face covering, but they are required to provide a reasonable accommodation if it does not cause an undue hardship. Businesses are encouraged to inform customers there are exceptions to the requirement that all individuals must wear a mask.\nThese frequently asked questions are to provide guidance regarding the application of the face covering requirement in Executive Order 2021-10 for businesses and other places of public accommodation subject to Article 5 of the Illinois Human Rights Act, 775 ILCS 5/. A: A face covering is a mask or cloth face covering that covers your nose and mouth.\nExceptions may be made for individuals with medical conditions or disabilities that prevent them from safely wearing a face covering. For more information, refer to the questions on reasonable accommodations. A: Masks still must be worn by everyone on planes, buses, trains, and other forms of public transportation; in transportation hubs, such as airports and train and bus stations; in health care settings; and in congregate facilities, such as correctional facilities and homeless shelters.\nThese frequently asked questions are to provide guidance regarding the application of the face covering requirement in Executive Order 2021-10 for businesses and other places of public accommodation subject to Article 5 of the Illinois Human Rights Act, 775 ILCS 5/. When Face Coverings are Required Q: What does it mean to wear a face covering?\nInformation 2:\nPublication date: 2013-08-06\nTitle: Illinois Strengthens Face Mask Rules in Businesses - WebMD\nContent:\nCOVID-19 is a new type of coronavirus that causes mild to severe cases. Here’s a quick guide on how to spot symptoms, risk factors, prevent spread of the disease, and find out what to do if you think you have it.\nInformation 3:\nPublication date: 2020-09-02\nTitle: Illinois Businesses Face Fines For Failing To Require Masks | WBEZ ...\nContent:\nPritzker’s new rule, approved by a 12-member state board Tuesday, would allow retailers to be fined up to $2,500 if they don’t require masks to be worn. ... URL Copied! WBEZ brings you fact-based news and information. Sign up for our newsletters to stay up to date on the stories that matter. A state panel Tuesday sided with Illinois Gov. JB Pritzker in his fight against COVID-19 by approving a rule that could mean fines of up to $2,500 for “rogue” businesses that don’t require patrons wear facial coverings.\nNow, the only consequences would be felt by businesses and their financial bottom lines. But that change wasn’t enough to satisfy Republicans, who failed in their effort to block the new rule from taking effect. The GOP pushback fell two votes shy of the necessary eight votes required to block the rule. “Honestly, having talked with some of the Illinoisans who refuse to wear masks, passing this administrative rule is not going to make people more likely to wear a mask,” said state Sen.\nSince then, Pritzker’s administration eliminated a provision from the earlier version of the rule that would have subjected individuals who run businesses that flout state mask-wearing requirements to a Class A misdemeanor, which theoretically could carry jail time upon conviction.\nLocal public health departments or law-enforcement agencies would first have to give rule-breaking businesses or groups written notice that they were violating the state mask-wearing order. Then, if that step didn’t produce compliance, local officials could order people in the business or facility to disperse.\nInformation 4:\nPublication date: 2023-04-14\nTitle: Mask Mandate FAQ - Illinois Manufacturers' Association\nContent:\nWhile the mask mandate applies to all employers, these new fines and penalties are aimed at consumer-facing businesses that are open to the public. Q: What businesses may be fined under this new rule? A: This new rule applies to any business, service, facility, or organization · open to the public. In these buildings open to the public, individuals should wear face coverings when unable to maintain a 6-foot social distance.\nA: This new rule applies to any business, service, facility, or organization · open to the public. In these buildings open to the public, individuals should wear face coverings when unable to maintain a 6-foot social distance. Generally speaking, manufacturing facilities are not open to the public and will not be significantly impacted by these new fines and penalties. Businesses may also be subject to penalties for hosting gatherings of more than 50 individuals.\nIn these buildings open to the public, individuals should wear face coverings when unable to maintain a 6-foot social distance. Generally speaking, manufacturing facilities are not open to the public and will not be significantly impacted by these new fines and penalties. Businesses may also be subject to penalties for hosting gatherings of more than 50 individuals. The specific language in the Administrative Rule provides “any businesses, service, facility or organization\nThe specific language in the Administrative Rule provides “any businesses, service, facility or organization · open to the public or employees shall require employees, customers, or other individuals who are over the age of two and able to medically tolerate a face covering to cover their nose and mouth with a face covering when on premises and unable to maintain a 6-foot social distance.\nInformation 5:\nPublication date: 2021-05-18\nTitle: Pritzker eases Illinois face covering rules, supports those still ...\nContent:\nBut they still call for wearing masks in crowded indoor settings — such as buses, planes, hospitals, prisons and homeless shelters. Under Pritzker’s updated rules, Illinoisans — whether vaccinated or not — will still be required to wear masks on public transportation and in transportation hubs, in correctional facilities and homeless shelters and in healthcare settings. Children will be required to wear masks in schools — those in daycare will also still be required to wear a face covering, according to a news release announcing the state’s changes.\nBut with the updated masking guidance, and the state now in the “bridge” phase of Pritzker’s reopening plan, the end of the pandemic could be near for Illinois. Gov. J.B. Pritzker puts on a mask during a briefing at the James R. Thompson Center in the Loop in November. ... The new CDC guidance announced last week allowed for fully vaccinated people to safely stop wearing masks outdoors and in the majority of indoor settings. But they still call for wearing masks in crowded indoor settings — such as buses, planes, hospitals, prisons and homeless shelters. Under Pritzker’s updated rules, Illinoisans — whether vaccinated or not — will still be required to wear masks on public transportation and in transportation hubs, in correctional facilities and homeless shelters and in healthcare settings.\nPritzker to ease Illinois rules on face coverings for fully vaccinated following CDC update · Shot of J&J for JB — Pritzker gets COVID-19 vaccine: ‘I feel great’ · As of Monday, the state has administered 10.4 million doses of the vaccine. A little over 4.8 million people in the state are fully vaccinated, according to data from the department of public health. The state’s updated policy on masks follows changes from the U.S.\n“With public health experts now saying fully vaccinated people can safely remove their masks in most settings, I’m pleased to follow the science and align Illinois’ policies with the CDC’s guidance.” · Pritzker went on in his statement to say he supports individuals and business who choose to continue to wear a mask “out of an abundance of caution as this pandemic isn’t over yet.” · The Democratic governor is one of those still wearing a mask — nearly eight weeks after receiving a dose of the Johnson & Johnson vaccine in Springfield. Asked about his face covering at an unrelated news conference on Monday, Pritzker said he thinks all residents “are going to have to adjust to the idea” of not needing to wear a mask.\n"},
    {'query': eval_prompt},
    ]

    resp_list = inference_vllm(
        mistral_engine, template, request_list, generation_config=generation_config, 
        # verbose=True, prompt_prefix="", output_prefix=""
    )

    for request, resp in zip(request_list, resp_list):
        result_mistral.append(resp['response'])


    # response_llama, history_llama = inference(model_mistral, template_mistral, eval_prompt)
    # result_mistral.append(response_llama)

        # 指定本地文件路径
file_path = '/home/qiuyang/workplace/swift/examples/pytorch/llm/my_tuning/data_siyuanzu/mistral_list7.json'

# 将数据写入本地文件
with open(file_path, 'w') as file:
    json.dump(result_mistral, file, indent=2)



