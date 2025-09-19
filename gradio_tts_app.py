import random
import io
from contextlib import redirect_stderr
from random import randint

import numpy as np
import torch
import gradio as gr
from src.chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
string_buf = None
last_audio_segments = []
last_sentences = []


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model


def batchify(l, batch_size):
    start_index = 0
    tot_len = 0
    for i, sent in enumerate(l):
        tot_len += len(sent)
        if i > 0 and tot_len >= batch_size:
            yield l[start_index:i]
            start_index = i
            tot_len = 0
    if start_index < len(l):
        yield l[start_index:]


def regenerate(model, sentence_indices, audio_prompt_path, exaggeration, temperature, cfgw, progress=gr.Progress()):
    global string_buf, last_audio_segments

    set_seed(randint(0, 2**32 - 1))
    sentences = [last_sentences[i] for i in sentence_indices]
    num_sentences = len(sentences)
    audio_segments = last_audio_segments
    progress((0, num_sentences), unit="Sentences", desc="Synthesizing speech...")
    progress_count = 0
    with io.StringIO() as string_buf, redirect_stderr(string_buf):
        for idx, sentence in zip(sentence_indices, sentences):
            wav = model.generate(
                sentence,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfgw,
            )
            #yield (model.sr, wav.squeeze(0).numpy())
            audio_segments[idx] = wav.squeeze(0).numpy()
            progress_count += 1
            progress((progress_count, num_sentences), unit="Sentences", desc="Synthesizing speech...")
    string_buf = None

    last_audio_segments = audio_segments
    return (model.sr, np.concatenate(audio_segments))


def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, progress=gr.Progress()):
    global string_buf, last_audio_segments, last_sentences
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))
    sentences = text.split(".")
    num_sentences = len(sentences)
    audio_segments = []
    batched_sentences = []
    progress((0, num_sentences), unit="Sentences", desc="Synthesizing speech...")
    progress_count = 0
    with io.StringIO() as string_buf, redirect_stderr(string_buf):
        for sentence_batch in batchify(sentences, 200):
            text_batch = ".".join(sentence_batch)
            wav = model.generate(
                text_batch,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfgw,
            )
            #yield (model.sr, wav.squeeze(0).numpy())
            audio_segments.append(wav.squeeze(0).numpy())
            progress_count += len(sentence_batch)
            progress((progress_count, num_sentences), unit="Sentences", desc="Synthesizing speech...")
            batched_sentences.append(text_batch)
    string_buf = None
    last_audio_segments = audio_segments
    last_sentences = batched_sentences
    return (model.sr, np.concatenate(audio_segments)), gr.Dropdown(choices=[(sent, i) for i, sent in enumerate(batched_sentences)], value=[], label="Chunks", multiselect=True, visible=True)


def get_tqdm_progress():
    if string_buf:
        ret_val = string_buf.getvalue().split('\r')[-1]
        string_buf.seek(0)
        string_buf.truncate()
        return ret_val.rstrip()
    else:
        return ""


def split_sentence(sentence_indices, split_mode):
    global last_sentences, last_audio_segments
    if len(sentence_indices) > 1:
        gr.Warning('Only a single sentence can be split!')
        return gr.Dropdown(choices=[(sent, i) for i, sent in enumerate(last_sentences)], value=[], label="Chunks", multiselect=True, visible=True)

    sentence_index = sentence_indices[0]
    split_func = str.split if split_mode == "Left" else str.rsplit
    splits = split_func(last_sentences[sentence_index].rstrip("."), ".", maxsplit=1)
    last_sentences = last_sentences[:sentence_index] + splits + last_sentences[sentence_index+1:]
    last_audio_segments = last_audio_segments[:sentence_index] + [None]*len(splits) + last_audio_segments[sentence_index+1:]
    return gr.Dropdown(choices=[(sent, i) for i, sent in enumerate(last_sentences)], value=list(range(sentence_index, sentence_index+len(splits))), label="Chunks", multiselect=True, visible=True)


def merge_sentences(sentence_indices):
    global last_sentences, last_audio_segments
    sorted_indices = sorted(sentence_indices)
    if np.diff(sorted_indices).max() > 1:
        gr.Warning('Only neighboring sentences can be merged!')
        return gr.Dropdown(choices=[(sent, i) for i, sent in enumerate(last_sentences)], value=[], label="Chunks", multiselect=True, visible=True)

    new_sentence = ".".join(last_sentences[sorted_indices[0]:sorted_indices[-1]+1])
    last_sentences = last_sentences[:sorted_indices[0]] + [new_sentence] + last_sentences[sorted_indices[-1]+1:]
    last_audio_segments = last_audio_segments[:sorted_indices[0]] + [None] + last_audio_segments[sorted_indices[-1]+1:]
    return gr.Dropdown(choices=[(sent, i) for i, sent in enumerate(last_sentences)], value=[sorted_indices[0]], label="Chunks", multiselect=True, visible=True)


with gr.Blocks(analytics_enabled=False) as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(value="What does the fox say?", label="Text to synthesize")
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")


        with gr.Column():
            with gr.Group():
                audio_output = gr.Audio(label="Output Audio") #, streaming=True, autoplay=True)
                progress_text = gr.Textbox(value=get_tqdm_progress, show_label=False, interactive=False,
                                       every=0.5)
            with gr.Row():
                regen_button = gr.Button("Re-generate")
                merge_button = gr.Button("Merge")
                with gr.Group():
                    split_button = gr.Button("Split")
                    split_lr_choice = gr.Radio(label="", choices=["Left", "Right"], value="Left")
            sentences_dropdown = gr.Dropdown(choices=[], value=[], label="Chunks", multiselect=True, visible=False)

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(lambda s: s.strip(), text, text).then(lambda: gr.Dropdown(choices=[], value=[],
                                      label="Chunks", multiselect=True, visible=False),
    None, sentences_dropdown).then(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output, sentences_dropdown]
    )

    regen_button.click(
        fn=regenerate,
        inputs=[
            model_state,
            sentences_dropdown,
            ref_wav,
            exaggeration,
            temp,
            cfg_weight,
        ],
        outputs=[audio_output]
    )

    merge_button.click(merge_sentences, sentences_dropdown, sentences_dropdown).then(
        fn=regenerate,
        inputs=[
            model_state,
            sentences_dropdown,
            ref_wav,
            exaggeration,
            temp,
            cfg_weight,
        ],
        outputs=[audio_output]
    )
    split_button.click(split_sentence, [sentences_dropdown, split_lr_choice], sentences_dropdown).then(
        fn=regenerate,
        inputs=[
            model_state,
            sentences_dropdown,
            ref_wav,
            exaggeration,
            temp,
            cfg_weight,
        ],
        outputs=[audio_output]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
