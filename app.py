import gradio as gr
import os
import torch
from torchvision import transforms
from lavis.processors import transforms_video
from lavis.datasets.data_utils import load_video_demo
from lavis.processors.blip_processors import ToUint8, ToTHWC
from lavis.models.sevila_models.sevila import SeViLA
from typing import Optional
import warnings
# model config
img_size = 224
num_query_token = 32
t5_model = 'google/flan-t5-xl'
drop_path_rate = 0
use_grad_checkpoint = False
vit_precision = "fp16"
freeze_vit = True
prompt = ''
max_txt_len = 77
answer_num = 5
apply_lemmatizer = False
task = 'freeze_loc_freeze_qa_vid'

# prompt
LOC_propmpt = 'Does the information within the frame provide the necessary details to accurately answer the given question?'
QA_prompt = 'Considering the information presented in the frame, select the correct answer from the options.'

# processors config
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)
image_size = img_size
transform = transforms.Compose([ToUint8(), ToTHWC(), transforms_video.ToTensorVideo(), normalize])

print('Model Loading \nLoading the SeViLA model can take a few minutes (typically 2-3).')
sevila = SeViLA(
    img_size=img_size,
    drop_path_rate=drop_path_rate,
    use_grad_checkpoint=use_grad_checkpoint,
    vit_precision=vit_precision,
    freeze_vit=freeze_vit,
    num_query_token=num_query_token,
    t5_model=t5_model,
    prompt=prompt,
    max_txt_len=max_txt_len,
    apply_lemmatizer=apply_lemmatizer,
    frame_num=4,
    answer_num=answer_num,
    task=task,
        )

sevila.load_checkpoint(url_or_filename='https://huggingface.co/Shoubin/SeViLA/resolve/main/sevila_pretrained.pth')
print('Model Loaded')

ANS_MAPPING = {0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E'}

# os.mkdir('video')

def sevila_demo(video, 
    question, 
    option1, option2, option3, 
    video_frame_num, 
    keyframe_num):
    
    if torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'
        
    global sevila 
    if device == "cpu":
        sevila = sevila.float()
    else:
        sevila = sevila.to(int(device))
        
    vpath = video 
    raw_clip, indice, fps, vlen = load_video_demo(
        video_path=vpath,
        n_frms=int(video_frame_num),
        height=image_size,
        width=image_size,
        sampling="uniform",
        clip_proposal=None
    )
    clip = transform(raw_clip.permute(1,0,2,3))
    clip = clip.float().to(int(device))
    clip = clip.unsqueeze(0)
    # check
    if option1[-1] != '.':
        option1 += '.'
    if option2[-1] != '.':
        option2 += '.' 
    if option3[-1] != '.':
        option3 += '.'
    option_dict = {0:option1, 1:option2, 2:option3}
    options = 'Option A:{} Option B:{} Option C:{}'.format(option1, option2, option3)
    text_input_qa = 'Question: ' + question + ' ' + options + ' ' + QA_prompt
    text_input_loc = 'Question: ' + question + ' ' + options + ' ' + LOC_propmpt
    
    out = sevila.generate_demo(clip, text_input_qa, text_input_loc, int(keyframe_num))
    # print(out)
    answer_id = out['output_text'][0]
    answer = option_dict[answer_id]
    select_index = out['frame_idx'][0]
    # images = [] 
    keyframes = []
    timestamps =[]
    
    # print('raw_clip', len(raw_clip))
    # for j in range(int(video_frame_num)):
    #     image = raw_clip[:, j, :, :].int()
    #     image = image.permute(1, 2, 0).numpy() 
    #     images.append(image)
    
    video_len = vlen/fps # seconds
    
    for i in select_index:
        image = raw_clip[:, i, :, :].int()
        image = image.permute(1, 2, 0).numpy() 
        keyframes.append(image)
        select_i = indice[i]
        time = round((select_i / vlen) * video_len, 2)
        timestamps.append(str(time)+'s')
    
    gr.components.Gallery(keyframes)
    #gr.components.Gallery(images)
    timestamps_des = ''
    for i in range(len(select_index)):
        timestamps_des += 'Keyframe {}: {} \n'.format(str(i+1), timestamps[i])
    
    return keyframes, timestamps_des, answer

with gr.Blocks(title="SeViLA demo") as demo:
    description = """<p style="text-align: center; font-weight: bold;">
        <span style="font-size: 28px">Self-Chained Image-Language Model for Video Localization and Question Answering</span>
        <br>
        <span style="font-size: 18px" id="author-info">
            <a href="https://yui010206.github.io/" target="_blank">Shoubin Yu</a>, 
            <a href="https://j-min.io/" target="_blank">Jaemin Cho</a>, 
            <a href="https://prateek-yadav.github.io/" target="_blank">Prateek Yadav</a>, 
            <a href="https://www.cs.unc.edu/~mbansal/" target="_blank">Mohit Bansal</a>
        </span> 
        <br>
        <span style="font-size: 18px" id="paper-info">
            [<a href="https://github.com/Yui010206/SeViLA" target="_blank">GitHub</a>]
            [<a href="https://arxiv.org/abs/2305.06988" target="_blank">Paper</a>]
        </span>
    </p>
    <p>
        To locate keyframes in a video and answer question, please:
        <br>
        (1) upolad your video; (2) write your question/options and set # video frame/# keyframe; (3) click Locate and Answer!
        <br>
        Just a heads up - loading the SeViLA model can take a few minutes (typically 2-3), and running examples requires about 12GB of memory.
        <br>
        We've got you covered! We've provided some example videos and questions below to help you get started. Feel free to try out SeViLA with these!
    </p>
    """
    gr.HTML(description)
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            video = gr.Video(label='Video') 
            question = gr.Textbox(placeholder="Why did the two ladies put their hands above their eyes while staring out?", label='Question')
            with gr.Row():
                option1 = gr.Textbox(placeholder="practicing cheer", label='Option 1')
                option2 = gr.Textbox(placeholder="posing for photo", label='Option 2')
                option3 = gr.Textbox(placeholder="to see better", label='Option 3')
            with gr.Row():
                video_frame_num = gr.Textbox(placeholder=32, label='# Video Frame')
                keyframe_num = gr.Textbox(placeholder=4, label='# Keyframe') 
            # device = gr.Textbox(placeholder=0, label='Device') 
            gen_btn = gr.Button(value='Locate and Answer!')
        with gr.Column(scale=1, min_width=600): 
            keyframes = gr.Gallery(
                label="Keyframes", show_label=False, elem_id="gallery",
                ).style(columns=[4], rows=[1], object_fit="contain", max_width=100, max_height=100)
            #keyframes = gr.Gallery(label='Keyframes')
            timestamps = gr.outputs.Textbox(label="Keyframe Timestamps")
            answer = gr.outputs.Textbox(label="Output Answer")
        
        gen_btn.click(
            sevila_demo,
            inputs=[video, question, option1, option2, option3, video_frame_num, keyframe_num],
            outputs=[keyframes, timestamps, answer],
            queue=True
        )
        #demo = gr.Interface(sevila_demo,
        #     inputs=[gr.Video(), question, option1, option2, option3, video_frame_num, keyframe_num, device],
        #     outputs=['gallery', timestamps, answer],
        #     examples=[['videos/demo1.mp4', 'Why did the two ladies put their hands above their eyes while staring out?', 'practicing cheer.', 'play ball.', 'to see better.', 32, 4, 0],
        #               ['videos/demo2.mp4', 'What did both of them do after completing skiing?', 'jump and pose.' , 'bend down.','raised their hands.', 32, 4, 0],
        #               ['videos/demo3.mp4', 'What room was Wilson breaking into when House found him?', 'the kitchen.' , 'the dining room.','the bathroom.', 32, 4, 0]]
        #     )
    with gr.Column():
        gr.Examples(
            inputs=[video, question, option1, option2, option3, video_frame_num, keyframe_num],
            outputs=[keyframes, timestamps, answer],
            fn=sevila_demo,
            examples=[['videos/demo1.mp4', 'Why did the two ladies put their hands above their eyes while staring out?', 'practicing cheer', 'to place wreaths', 'to see better', 32, 4],
                      ['videos/demo2.mp4', 'What did both of them do after completing skiing?', 'jump and pose' , 'bend down','raised their hands', 32, 4],
                      ['videos/demo3.mp4', 'What room was Wilson breaking into when House found him?', 'the bedroom' , 'the bathroom','the kitchen', 32, 4]],
            cache_examples=False,
        )
demo.queue(concurrency_count=1, api_open=False)          
demo.launch(share=False) 