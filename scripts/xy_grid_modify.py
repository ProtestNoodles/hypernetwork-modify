from collections import namedtuple
from copy import copy
from itertools import permutations, chain
import random
import csv
from io import StringIO
from PIL import Image
import numpy as np

import modules.scripts as scripts
import gradio as gr

from modules import images, paths, sd_samplers, processing, sd_models
from modules.hypernetworks import hypernetwork
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae
import glob
import os
import re
import sys
import traceback

from scripts.hn_modify import modify_hypernetwork

hn_operands = ["(ws)", "(+)", "(-)", "(ad)"]


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")

    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_order(p, x, xs):
    token_order = []

    # Initally grab the tokens from the prompt, so they can be replaced in order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))

    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]

    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def apply_sampler(p, x, xs):
    sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if sampler_name is None:
        raise RuntimeError(f"Unknown sampler: {x}")

    p.sampler_name = sampler_name


def confirm_samplers(p, xs):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            raise RuntimeError(f"Unknown sampler: {x}")


def apply_checkpoint(p, x, xs):
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    modules.sd_models.reload_model_weights(shared.sd_model, info)
    p.sd_model = shared.sd_model


def confirm_checkpoints(p, xs):
    for x in xs:
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def apply_hypernetwork(p, x, xs):
    if x.lower() in ["", "none"]:
        name = None
        hypernetwork.load_hypernetwork(name)
    else:
        # name = hypernetwork.find_closest_hypernetwork_name(x)
        # if not name:
        #     raise RuntimeError(f"Unknown hypernetwork: {x}")
        op = None
        hn_1, hn_2, hn_3 = None, None, None
        for operand in hn_operands:
            if operand in x:
                op = operand
                x = x.split(operand)
        if type(x) is str:
            hn_1 = x
        else:
            hn_1 = x[0]
            hn_2 = x[1]
            if op == "(ad)":
                hn_3 = x[2]

        kwargs = {"save_file": False}
        o_dict = {"(ws)": "Weighted Sum", "(+)": "Add", "(-)": "Subtract", "(ad)": "Add Difference"}
        hn_1_opts_d = {"320": "hn_1_320", "640": "hn_1_640", "768": "hn_1_768", "1024": "hn_1024", "1280": "hn_1_1280"}
        hn_2_opts_d = {"320": "hn_2_320", "640": "hn_2_640", "768": "hn_2_768", "1024": "hn_2_1024",
                       "1280": "hn_2_1280"}
        arg_dict = {"af": "activation_func_strategy", "mms": "missing_module_strategy",
                    "lts": "ls_tensor_resize_strategy",
                    "wsr": "ws_ratio", "str": "multiplier", "save": "save_file"}

        if op:
            kwargs["merge_method"] = o_dict[op]

        if "[" in hn_1 and "]" in hn_1:
            hn_1_opts = hn_1.split("[")[1].split("]")[0].split(";")
            for opt in hn_1_opts:
                if ":" in opt:
                    arg = opt.split(":")[0]
                    val = opt.split(":")[1]
                    if arg in hn_1_opts_d:
                        kwargs[hn_1_opts_d[arg]] = float(val)
                    else:
                        raise RuntimeError(f"Unknown argument: {arg}")
            hn_1 = re.sub(r"\[.+\]", "", hn_1)

        if "{" in hn_1 and "}" in hn_1:
            arg_opts = hn_1.split("{")[1].split("}")[0].split(";")
            for opt in arg_opts:
                if ":" in opt:
                    arg = opt.split(":")[0]
                    val = opt.split(":")[1]
                    if arg in ["wsr", "str"]:
                        val = float(val)
                    if arg == "save":
                        kwargs["save_file"] = True
                        if val != "auto":
                            kwargs["custom_name"] = val
                    if arg in arg_dict:
                        kwargs[arg_dict[arg]] = val
                    else:
                        raise RuntimeError(f"Unknown argument: {arg}")
            hn_1 = re.sub(r"\{.+\}", "", hn_1)

        if ":" in hn_1:
            kwargs["hn_1_strength"] = float(hn_1.split(":")[-1])
            hn_1 = hn_1.split(":")[0]

        hn_1 = hypernetwork.find_closest_hypernetwork_name(hn_1)
        # print(f"\n\n------\n{hn_1}\n{shared.hypernetworks[hn_1]}------\n\n")

        # yeah i know duplicated code is bad, i should refactor this
        if hn_2:
            if "[" in hn_2 and "]" in hn_2:
                hn_2_opts = hn_2.split("[")[1].split("]")[0].split(";")
                for opt in hn_2_opts:
                    if ":" in opt:
                        arg = opt.split(":")[0]
                        val = opt.split(":")[1]
                        if arg in hn_2_opts_d:
                            kwargs[hn_2_opts_d[arg]] = float(val)
                        else:
                            raise RuntimeError(f"Unknown argument: {arg}")
                hn_2 = re.sub(r"\[.+\]", "", hn_2)

            if "{" in hn_2 and "}" in hn_2:
                arg_opts = hn_2.split("{")[1].split("}")[0].split(";")
                for opt in arg_opts:
                    if ":" in opt:
                        arg = opt.split(":")[0]
                        val = opt.split(":")[1]
                        if arg in ["wsr", "str"]:
                            val = float(val)
                        if arg == "save":
                            kwargs["save_file"] = True
                            if val != "auto":
                                kwargs["custom_name"] = val
                        if arg in arg_dict:
                            kwargs[arg_dict[arg]] = val
                        else:
                            raise RuntimeError(f"Unknown argument: {arg}")
                hn_2 = re.sub(r"\{.+\}", "", hn_2)

            if ":" in hn_2:
                kwargs["hn_2_strength"] = float(hn_2.split(":")[-1])
                hn_2 = hn_2.split(":")[0]

            kwargs["hn_2_name"] = hypernetwork.find_closest_hypernetwork_name(hn_2)

        if hn_3:
            if "{" in hn_3 and "}" in hn_3:
                arg_opts = hn_3.split("{")[1].split("}")[0].split(";")
                for opt in arg_opts:
                    if ":" in opt:
                        arg = opt.split(":")[0]
                        val = opt.split(":")[1]
                        if arg in ["wsr", "str"]:
                            val = float(val)
                        if arg == "save":
                            kwargs["save_file"] = True
                            if val != "auto":
                                kwargs["custom_name"] = val
                        if arg in arg_dict:
                            kwargs[arg_dict[arg]] = val
                        else:
                            raise RuntimeError(f"Unknown argument: {arg}")
                hn_3 = re.sub(r"\{.+\}", "", hn_3)

            hn_3 = hypernetwork.find_closest_hypernetwork_name(hn_3)
            kwargs["hn_3_name"] = hn_3

        kwargs["return_hn"] = True
        # kwargs["save_file"] = False
        print(hn_1, kwargs)
        sd_1 = modify_hypernetwork(hn_1, **kwargs)
        load_modified_hypernetwork(sd_1[0])


def parse_dropout_structure(layer_structure, use_dropout, last_layer_dropout):
    if layer_structure is None:
        layer_structure = [1, 2, 1]
    if not use_dropout:
        return [0] * len(layer_structure)
    dropout_values = [0]
    dropout_values.extend([0.3] * (len(layer_structure) - 3))
    if last_layer_dropout:
        dropout_values.append(0.3)
    else:
        dropout_values.append(0)
    dropout_values.append(0)
    return dropout_values


class ModifiedHypernetwork(hypernetwork.Hypernetwork):
    def load(self, state_dict):
        self.filename = state_dict["name"] + ".pt"
        self.name = state_dict["name"]

        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        print(self.layer_structure)
        optional_info = state_dict.get('optional_info', None)
        if optional_info is not None:
            print(f"INFO:\n {optional_info}\n")
            self.optional_info = optional_info
        self.activation_func = state_dict.get('activation_func', None)
        print(f"Activation function is {self.activation_func}")
        self.weight_init = state_dict.get('weight_initialization', 'Normal')
        print(f"Weight initialization is {self.weight_init}")
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        print(f"Layer norm is set to {self.add_layer_norm}")
        self.dropout_structure = state_dict.get('dropout_structure', None)
        self.use_dropout = True if self.dropout_structure is not None and any(self.dropout_structure) else state_dict.get('use_dropout', False)
        print(f"Dropout usage is set to {self.use_dropout}" )
        self.activate_output = state_dict.get('activate_output', True)
        print(f"Activate last layer is set to {self.activate_output}")
        self.last_layer_dropout = state_dict.get('last_layer_dropout', False)  # Silent fix for HNs before 4918eb6
        # Dropout structure should have same length as layer structure, Every digits should be in [0,1), and last digit must be 0.
        if self.dropout_structure is None:
            print("Using previous dropout structure")
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)
        print(f"Dropout structure is set to {self.dropout_structure}")

        optimizer_saved_dict = torch.load(self.filename + '.optim', map_location = 'cpu') if os.path.exists(self.filename + '.optim') else {}
        self.optimizer_name = "AdamW"

        if sd_models.model_hash(self.filename) == optimizer_saved_dict.get('hash', None):
            self.optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)
        else:
            self.optimizer_state_dict = None
        if self.optimizer_state_dict:
            self.optimizer_name = optimizer_saved_dict.get('optimizer_name', 'AdamW')
            print("Loaded existing optimizer from checkpoint")
            print(f"Optimizer name is {self.optimizer_name}")
        else:
            print("No saved optimizer exists in checkpoint")

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (
                    hypernetwork.HypernetworkModule(size, sd[0], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                    hypernetwork.HypernetworkModule(size, sd[1], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                )

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)
        self.eval()


def load_modified_hypernetwork(sd):
    if sd:
        print(f"Loading hypernetwork {sd['name']}")
        try:
            shared.loaded_hypernetwork = ModifiedHypernetwork()
            shared.loaded_hypernetwork.load(sd)

        except Exception:
            print(f"Error loading hypernetwork {sd['name']}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    else:
        if shared.loaded_hypernetwork is not None:
            print("Unloading hypernetwork")

        shared.loaded_hypernetwork = None


def apply_hypernetwork_strength(p, x, xs):
    hypernetwork.apply_strength(x)


def confirm_hypernetworks(p, xs):
    for x in xs:
        if x.lower() in ["", "none"]:
            continue
        for operand in hn_operands:
            if operand in x:
                x = x.split(operand)
        if type(x) is str:
            x = [x]

        for y in x:
            y = re.sub(r"\[.+\]", "", y)
            y = re.sub(r"\{.+\}", "", y)
            y = y.split(":")[0]
            if not hypernetwork.find_closest_hypernetwork_name(y):
                raise RuntimeError(f"Unknown hypernetwork: {y}")


def apply_clip_skip(p, x, xs):
    opts.data["CLIP_stop_at_last_layers"] = x


def apply_upscale_latent_space(p, x, xs):
    if x.lower().strip() != '0':
        opts.data["use_scale_latent_for_hires_fix"] = True
    else:
        opts.data["use_scale_latent_for_hires_fix"] = False


def find_vae(name: str):
    if name.lower() in ['auto', 'none']:
        return name
    else:
        vae_path = os.path.abspath(os.path.join(paths.models_path, 'VAE'))
        found = glob.glob(os.path.join(vae_path, f'**/{name}.*pt'), recursive=True)
        if found:
            return found[0]
        else:
            return 'auto'


def apply_vae(p, x, xs):
    if x.lower().strip() == 'none':
        modules.sd_vae.reload_vae_weights(shared.sd_model, vae_file='None')
    else:
        found = find_vae(x)
        if found:
            v = modules.sd_vae.reload_vae_weights(shared.sd_model, vae_file=found)


def apply_styles(p: StableDiffusionProcessingTxt2Img, x: str, _):
    p.styles = x.split(',')


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)

    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def do_nothing(p, x, xs):
    pass


def format_nothing(p, opt, x):
    return ""


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x


AxisOption = namedtuple("AxisOption", ["label", "type", "apply", "format_value", "confirm"])
AxisOptionImg2Img = namedtuple("AxisOptionImg2Img", ["label", "type", "apply", "format_value", "confirm"])

axis_options = [
    AxisOption("Nothing", str, do_nothing, format_nothing, None),
    AxisOption("Seed", int, apply_field("seed"), format_value_add_label, None),
    AxisOption("Var. seed", int, apply_field("subseed"), format_value_add_label, None),
    AxisOption("Var. strength", float, apply_field("subseed_strength"), format_value_add_label, None),
    AxisOption("Steps", int, apply_field("steps"), format_value_add_label, None),
    AxisOption("CFG Scale", float, apply_field("cfg_scale"), format_value_add_label, None),
    AxisOption("Prompt S/R", str, apply_prompt, format_value, None),
    AxisOption("Prompt order", str_permutations, apply_order, format_value_join_list, None),
    AxisOption("Sampler", str, apply_sampler, format_value, confirm_samplers),
    AxisOption("Checkpoint name", str, apply_checkpoint, format_value, confirm_checkpoints),
    AxisOption("Hypernetwork", str, apply_hypernetwork, format_value, confirm_hypernetworks),
    AxisOption("Hypernet str.", float, apply_hypernetwork_strength, format_value_add_label, None),
    AxisOption("Sigma Churn", float, apply_field("s_churn"), format_value_add_label, None),
    AxisOption("Sigma min", float, apply_field("s_tmin"), format_value_add_label, None),
    AxisOption("Sigma max", float, apply_field("s_tmax"), format_value_add_label, None),
    AxisOption("Sigma noise", float, apply_field("s_noise"), format_value_add_label, None),
    AxisOption("Eta", float, apply_field("eta"), format_value_add_label, None),
    AxisOption("Clip skip", int, apply_clip_skip, format_value_add_label, None),
    AxisOption("Denoising", float, apply_field("denoising_strength"), format_value_add_label, None),
    AxisOption("Hires upscaler", str, apply_field("hr_upscaler"), format_value_add_label, None),
    AxisOption("Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight"), format_value_add_label, None),
    AxisOption("VAE", str, apply_vae, format_value_add_label, None),
    AxisOption("Styles", str, apply_styles, format_value_add_label, None),
]


def draw_xy_grid(p, xs, ys, x_labels, y_labels, cell, draw_legend, include_lone_images):
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]

    # Temporary list of all the images that are generated to be populated into the grid.
    # Will be filled with empty images for any individual step that fails to process properly
    image_cache = []

    processed_result = None
    cell_mode = "P"
    cell_size = (1, 1)

    state.job_count = len(xs) * len(ys) * p.n_iter

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state.job = f"{ix + iy * len(xs) + 1} out of {len(xs) * len(ys)}"

            processed: Processed = cell(x, y)
            try:
                # this dereference will throw an exception if the image was not processed
                # (this happens in cases such as if the user stops the process from the UI)
                processed_image = processed.images[0]

                if processed_result is None:
                    # Use our first valid processed result as a template container to hold our full results
                    processed_result = copy(processed)
                    cell_mode = processed_image.mode
                    cell_size = processed_image.size
                    processed_result.images = [Image.new(cell_mode, cell_size)]

                image_cache.append(processed_image)
                if include_lone_images:
                    processed_result.images.append(processed_image)
                    processed_result.all_prompts.append(processed.prompt)
                    processed_result.all_seeds.append(processed.seed)
                    processed_result.infotexts.append(processed.infotexts[0])
            except:
                image_cache.append(Image.new(cell_mode, cell_size))

    if not processed_result:
        print("Unexpected error: draw_xy_grid failed to return even a single processed image")
        return Processed()

    grid = images.image_grid(image_cache, rows=len(ys))
    if draw_legend:
        grid = images.draw_grid_annotations(grid, cell_size[0], cell_size[1], hor_texts, ver_texts)

    processed_result.images[0] = grid

    return processed_result


class SharedSettingsStackHelper(object):
    def __enter__(self):
        self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        self.hypernetwork = opts.sd_hypernetwork
        self.model = shared.sd_model
        self.vae = opts.sd_vae

    def __exit__(self, exc_type, exc_value, tb):
        modules.sd_models.reload_model_weights(self.model)
        modules.sd_vae.reload_vae_weights(self.model, vae_file=find_vae(self.vae))

        hypernetwork.load_hypernetwork(self.hypernetwork)
        hypernetwork.apply_strength()

        opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(
    r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*\])?\s*")
re_range_count_float = re.compile(
    r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*\])?\s*")


class Script(scripts.Script):
    def title(self):
        return "HN Modify Plot"

    def ui(self, is_img2img):
        current_axis_options = [x for x in axis_options if
                                type(x) == AxisOption or type(x) == AxisOptionImg2Img and is_img2img]

        with gr.Row():
            x_type = gr.Dropdown(label="X type", choices=[x.label for x in current_axis_options],
                                 value=current_axis_options[10].label, type="index", elem_id=self.elem_id("x_type"))
            x_values = gr.Textbox(label="X values", lines=1, elem_id=self.elem_id("x_values"))

        with gr.Row():
            y_type = gr.Dropdown(label="Y type", choices=[x.label for x in current_axis_options],
                                 value=current_axis_options[0].label, type="index", elem_id=self.elem_id("y_type"))
            y_values = gr.Textbox(label="Y values", lines=1, elem_id=self.elem_id("y_values"))

        draw_legend = gr.Checkbox(label='Draw legend', value=True, elem_id=self.elem_id("draw_legend"))
        include_lone_images = gr.Checkbox(label='Include Separate Images', value=False,
                                          elem_id=self.elem_id("include_lone_images"))
        no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=False, elem_id=self.elem_id("no_fixed_seeds"))

        # HN Modify collapsible info text
        with gr.Accordion(label="XY Hypernetwork Modify Help", open=False):
            helptext = gr.HTML(
                """
                <p><h1>Hi There! <img style="display: inline; vertical-align: text-bottom;" src="file=extensions/hypernetwork-modify/res/llama-up-and-down.gif"/></h1>
                <br>This is a modified version of the X/Y plot script that allows you to quickly test hypernetwork settings
                <br>To use this, select [Hypernetwork] in x or y and type the desired settings in the textbox
                <br>
                <br>The syntax is: hypernetwork_name[hypernetwork_option;second_option]:multiplier(optional_operation)second_hypernetwork{overall setting}
                <br>For example: <b>anime[640:2;320:0]</b>
                <br>This will use the anime hypernetwork with the 640 layer at 2x strength and the 320 layer at 0x strength
                <br>Another example: <b>anime(ws)furry_3{wsr:0.2}</b>
                <br>That will use a weighted sum with a ratio of 0.2 for the anime network and 0.8 for the furry_3 network
                <br>Here's one more: <b>anime[1280:0](+)furry_3[320:0;640:0;768:0]</b>
                <br>This will essentially replace the anime 1280 layer with the furry_3 1280 layer
                <br>One last example: <b>anime(-)furry_3:0.5</b>
                <br>This will subtract the furry network at 0.5 strength from the anime network
                <br>
                <br>Here's a cheat sheet for the available options, you can cross-reference these with the main tab:
                <br>To change the strength of a network you can add a :2 or :0.5 at the end on a network, eg anime:2
                <br>The Operands are:
                <br>(ws) = Weighted Sum, (+) = Add, (-) = Subtract, (ad) = Add Difference, (you will have to place (ad) between all three)
                <br>The Individual Network settings, to be used with [] angle brackets, are:
                <br>320, 640, 768, 1280 = individual layer strength, eg anime[320:2] will set the 320 layer to 2x strength
                <br>The overall settings, to be used with {} curly brackets, are:
                <br>wsr = Weighted Sum / Add Difference Ratio, af = Activation Function Strategy, mms = Missing Module Strategy,
                <br>lts = Layer Structure/ Tensor Resize Strategy, str = Overall Strength of resulting network,
                <br>save = Save the resulting network under specified name, eg anime{save:anime2} will save the resulting network as anime2
                <br>unless you enter auto, in that case the network will autogen the name
                <br>multiple settings should be seperated by a semicolon ; since the comma , is already in use
                <br>
                <br>Hopefully this isn't too confusing <img style="display: inline; vertical-align: text-bottom;" src="file=extensions/hypernetwork-modify/res/llama-depress.gif"/>, 
                refer to the main tab for more information!</p>
                """)

        return [x_type, x_values, y_type, y_values, draw_legend, include_lone_images, no_fixed_seeds]

    def run(self, p, x_type, x_values, y_type, y_values, draw_legend, include_lone_images, no_fixed_seeds):
        if not no_fixed_seeds:
            modules.processing.fix_seed(p)

        if not opts.return_grid:
            p.batch_size = 1

        def process_axis(opt, vals):
            if opt.label == 'Nothing':
                return [0]

            valslist = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(vals)))]

            if opt.type == int:
                valslist_ext = []

                for val in valslist:
                    m = re_range.fullmatch(val)
                    mc = re_range_count.fullmatch(val)
                    if m is not None:
                        start = int(m.group(1))
                        end = int(m.group(2)) + 1
                        step = int(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += list(range(start, end, step))
                    elif mc is not None:
                        start = int(mc.group(1))
                        end = int(mc.group(2))
                        num = int(mc.group(3)) if mc.group(3) is not None else 1

                        valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == float:
                valslist_ext = []

                for val in valslist:
                    m = re_range_float.fullmatch(val)
                    mc = re_range_count_float.fullmatch(val)
                    if m is not None:
                        start = float(m.group(1))
                        end = float(m.group(2))
                        step = float(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += np.arange(start, end + step, step).tolist()
                    elif mc is not None:
                        start = float(mc.group(1))
                        end = float(mc.group(2))
                        num = int(mc.group(3)) if mc.group(3) is not None else 1

                        valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == str_permutations:
                valslist = list(permutations(valslist))

            valslist = [opt.type(x) for x in valslist]

            # Confirm options are valid before starting
            if opt.confirm:
                opt.confirm(p, valslist)

            return valslist

        x_opt = axis_options[x_type]
        xs = process_axis(x_opt, x_values)

        y_opt = axis_options[y_type]
        ys = process_axis(y_opt, y_values)

        def fix_axis_seeds(axis_opt, axis_list):
            if axis_opt.label in ['Seed', 'Var. seed']:
                return [int(random.randrange(4294967294)) if val is None or val == '' or val == -1 else val for val in
                        axis_list]
            else:
                return axis_list

        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs)
            ys = fix_axis_seeds(y_opt, ys)

        if x_opt.label == 'Steps':
            total_steps = sum(xs) * len(ys)
        elif y_opt.label == 'Steps':
            total_steps = sum(ys) * len(xs)
        else:
            total_steps = p.steps * len(xs) * len(ys)

        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
            total_steps *= 2

        print(
            f"X/Y plot will create {len(xs) * len(ys) * p.n_iter} images on a {len(xs)}x{len(ys)} grid. (Total steps to process: {total_steps * p.n_iter})")
        shared.total_tqdm.updateTotal(total_steps * p.n_iter)

        grid_infotext = [None]

        def cell(x, y):
            pc = copy(p)
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)

            res = process_images(pc)

            if grid_infotext[0] is None:
                pc.extra_generation_params = copy(pc.extra_generation_params)

                if x_opt.label != 'Nothing':
                    pc.extra_generation_params["X Type"] = x_opt.label
                    pc.extra_generation_params["X Values"] = x_values
                    if x_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed X Values"] = ", ".join([str(x) for x in xs])

                if y_opt.label != 'Nothing':
                    pc.extra_generation_params["Y Type"] = y_opt.label
                    pc.extra_generation_params["Y Values"] = y_values
                    if y_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                        pc.extra_generation_params["Fixed Y Values"] = ", ".join([str(y) for y in ys])

                grid_infotext[0] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)

            return res

        with SharedSettingsStackHelper():
            processed = draw_xy_grid(
                p,
                xs=xs,
                ys=ys,
                x_labels=[x_opt.format_value(p, x_opt, x) for x in xs],
                y_labels=[y_opt.format_value(p, y_opt, y) for y in ys],
                cell=cell,
                draw_legend=draw_legend,
                include_lone_images=include_lone_images
            )

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "xy_grid", info=grid_infotext[0],
                              extension=opts.grid_format, prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        return processed
