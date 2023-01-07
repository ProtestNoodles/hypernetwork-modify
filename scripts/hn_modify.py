import os
import sys
import time
import re
import torch
import glob
import math
import gradio as gr
from modules import shared, sd_models, script_callbacks, ui, hypernetworks

# from modules.hypernetworks import hypernetwork


possible_layers = [320, 640, 768, 1024, 1280]


def fix_old_state_dict(hn):
    # changes the keys from the old NAI format to the new format
    changes = {
        'linear1.weight': 'linear.0.weight',
        'linear1.bias': 'linear.0.bias',
        'linear2.weight': 'linear.1.weight',
        'linear2.bias': 'linear.1.bias',
    }

    for layer in hn:
        if type(layer) == int and layer in possible_layers:
            for it in hn[layer]:
                for fr, to in changes.items():
                    x = it.get(fr, None)
                    if x is None:
                        continue

                    del it[fr]
                    it[to] = x

    return hn


def get_layers(network):
    layers = []
    for size, sd in network.items():
        if type(size) == int and size in possible_layers:
            layers.append(size)
    return layers


def get_layer_structure(network, calculate=False):
    if not calculate:
        if "layer_structure" in network:
            return network["layer_structure"]

    # old hypernets don't have the layer_structure key, so we have to calculate it
    # there's probably a better way to do this, but this works
    layers = get_layers(network)
    layer_structure = []
    for w in network[layers[0]][0]:
        if w.endswith("weight"):
            layer_structure.append(network[layers[0]][0][w].shape[0] / network[layers[0]][0][w].shape[1])

    layer_structure.pop(-1)

    for i in range(1, len(layer_structure)):
        layer_structure[i] = layer_structure[i] * layer_structure[i - 1]
        layer_structure[i] = float(round(layer_structure[i], 1))

    layer_structure = [1.0] + layer_structure + [1.0]

    return layer_structure


def modify_hypernetwork(hn_1_name, hn_2_name=None, hn_3_name=None,
                        hn_1_strength=1.0, hn_1_320=1.0, hn_1_640=1.0, hn_1_768=1.0, hn_1_1024=1.0, hn_1_1280=1.0,
                        hn_2_strength=1.0, hn_2_320=1.0, hn_2_640=1.0, hn_2_768=1.0, hn_2_1024=1.0, hn_2_1280=1.0,
                        merge_method="Weighted Sum", multiplier=1.0, custom_name=None,
                        missing_module_strategy="Add", ls_tensor_resize_strategy="Extend", activation_func_strategy="Keep Primary", ws_ratio=0.5,
                        save_file=True, return_hn=False):
    if not hn_1_name or hn_1_name == "None":
        return "ERROR:\nPrimary network is required!"

    return_text = ""

    # opt dict conversion because i used a dict before i figured out that gradio doesn't support it
    opt_1 = {"multiplier": hn_1_strength, 320: hn_1_320, 640: hn_1_640, 768: hn_1_768, 1024: hn_1_1024, 1280: hn_1_1280}
    opt_2 = {"multiplier": hn_2_strength, 320: hn_2_320, 640: hn_2_640, 768: hn_2_768, 1024: hn_2_1024, 1280: hn_2_1280}

    hn_1 = shared.hypernetworks.get(hn_1_name, None)

    sd_1 = torch.load(hn_1, map_location='cpu')
    sd_1 = fix_old_state_dict(sd_1)
    sd_1_layers = get_layers(sd_1)
    sd_1_structure = get_layer_structure(sd_1)
    if activation_func_strategy not in ["Keep Primary", "Keep Secondary"]:
        sd_1["activation_func"] = activation_func_strategy

    if hn_2_name and hn_2_name != "None":
        hn_2 = shared.hypernetworks.get(hn_2_name, None)
        sd_2 = torch.load(hn_2, map_location='cpu')
        sd_2 = fix_old_state_dict(sd_2)
        sd_2_layers = get_layers(sd_2)
        sd_2_structure = get_layer_structure(sd_2)
        if activation_func_strategy == "Keep Secondary":
            if "activation_func" in sd_2:
                sd_1["activation_func"] = sd_2["activation_func"]
            else:
                return_text += "WARNING: Secondary network does not have an activation function, so activation will be Primary!\n"

        if len(sd_1_layers) != len(sd_2_layers):
            return_text += f"WARNING: Second network has different number of layers!\nNetwork 1: {sd_1_layers}, Network 2: {sd_2_layers}\n"
        if sd_1_structure != sd_2_structure:
            return_text += f"WARNING: Second network has different layer structure!\nNetwork 1:{sd_1_structure}, Network 2: {sd_2_structure}\n"

    if hn_3_name and hn_3_name != "None":
        if merge_method != "Add Difference":
            return "ERROR:\nThird network can only be used with 'Add Difference' method!"

        hn_3 = shared.hypernetworks.get(hn_3_name, None)
        sd_3 = torch.load(hn_3, map_location='cpu')
        sd_3 = fix_old_state_dict(sd_3)
        sd_3_layers = get_layers(sd_3)
        sd_3_structure = get_layer_structure(sd_3)

        if len(sd_1_layers) != len(sd_3_layers):
            return_text += f"WARNING: Third network has different number of layers!\nNetwork 1: {sd_1_layers}, Network 3: {sd_3_layers}\n"
        if sd_1_structure != sd_3_structure:
            return_text += f"WARNING: Third network has different layer structure!\nNetwork 1:{sd_1_structure}, Network 3: {sd_3_structure}\n"

    # merge the two networks
    # the network structure is:
    # sd1
    # ----> 768 (Tuple)
    # --------> 0 (OrderedDict)
    # ------------> "linear.0.weight", Tensor
    # ------------> "linear.0.bias", Tensor
    # ...
    # ...
    # --------> 1 (OrderedDict)
    # ------------> "linear.0.weight", Tensor
    # ...
    # ...
    # ----> 320 (Tuple)
    # ...
    # ...
    # ----> "layer structure"
    # ----> "activation func"
    # ... etc
    # sorry that this is such a mess, i'm a dummy uwu
    for layer in possible_layers:
        if hn_2_name and hn_2_name != "None":
            if missing_module_strategy == "Add":
                # print("adding")
                if layer not in sd_1_layers and layer in sd_2_layers:
                    # print("adding layer to primary")
                    sd_1[layer] = sd_2.get(layer)
                    opt_1[layer] = opt_2[layer]
                    # del sd_2[layer]
                if layer not in sd_2_layers and layer in sd_1_layers:
                    # print("adding layer to secondary")
                    sd_2[layer] = sd_1.get(layer)
            elif missing_module_strategy == "Remove":
                if layer in sd_1_layers and layer not in sd_2_layers:
                    # print(f"Removing layer {layer} from primary network!")
                    del sd_1[layer]
                    continue
        if layer not in sd_1:
            continue

        for d_it in range(len(sd_1[layer])):
            d = sd_1[layer][d_it]
            if hn_2_name and hn_2_name != "None":
                d_2 = sd_2[layer][d_it]
                # print(len(d), len(d_2))
                if len(d) != len(d_2):
                    del_later = []
                    for k in d_2.keys():
                        # print(k)
                        if k not in d:
                            if ls_tensor_resize_strategy == "Extend":
                                # return_text += f"Extending layer {layer} in primary network!\n"
                                d[k] = d_2[k]
                            else:
                                del_later.append(k)
                    for k2 in del_later:
                        del d_2[k2]

                    del_later = []
                    for k in d.keys():

                        if k not in d_2:
                            if ls_tensor_resize_strategy == "Extend":
                                d_2[k] = d[k]
                            else:
                                del_later.append(k)
                    for k2 in del_later:
                        del d[k2]

            for t_it in range(len(d)):
                t = [k for k in d][t_it]
                if type(d[t]) == torch.Tensor:
                    d[t] = d[t] * opt_1[layer] * opt_1["multiplier"]

                    if hn_2_name and hn_2_name != "None":
                        if layer in sd_2_layers:
                            d_2[t] = list(d_2.items())[t_it][-1] * opt_2[layer] * opt_2["multiplier"]

                            if d[t].shape != d_2[t].shape:
                                # print("layer size mismatch")
                                if sum(sd_1_structure) > sum(sd_2_structure):
                                    if ls_tensor_resize_strategy == "Extend":
                                        d_2[t] = d_2[t].resize_as_(d[t])
                                    elif ls_tensor_resize_strategy == "Shrink":
                                        d[t] = d[t].resize_as_(d_2[t])
                                else:
                                    if ls_tensor_resize_strategy == "Extend":
                                        d[t] = d[t].resize_as_(d_2[t])
                                    elif ls_tensor_resize_strategy == "Shrink":
                                        d_2[t] = d_2[t].resize_as_(d[t])

                            if merge_method == "Weighted Sum":
                                d[t] = d[t] * ws_ratio + d_2[t] * (1 - ws_ratio)
                            if merge_method == "Add":
                                d[t] = d[t] + d_2[t]
                            elif merge_method == "Subtract":
                                d[t] = d[t] - d_2[t]
                            elif merge_method == "Multiply":
                                d[t] = d[t] * d_2[t]

                            if merge_method == "Add Difference":
                                if layer not in sd_3_layers:
                                    return_text += f"ERROR: Third network is missing layer {layer}!\n"
                                    continue
                                hn_3_tensor = list(sd_3[layer][d_it].items())[t_it][-1]
                                d[t] = d[t] + (d_2[t] - hn_3_tensor) * ws_ratio

                    d[t] = d[t] * math.sqrt(multiplier)

    sd_1["layer_structure"] = get_layer_structure(sd_1, calculate=True)
    if "dropout_structure" in sd_1:
        if len(sd_1["dropout_structure"]) != len(sd_1["layer_structure"]):
            dr_replace = 0
            if sd_1["use_dropout"]:
                dr_replace = sd_1["dropout_structure"][1]
            sd_1["dropout_structure"] = [0] + [dr_replace] * (len(sd_1["layer_structure"]) - 2) + [0]

    if save_file:
        folder_path = os.path.join(shared.cmd_opts.hypernetwork_dir, "~modified")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if custom_name:
            name = custom_name
        else:
            name = re.sub(r"\(.+\)", "", hn_1_name)
            for k, v in opt_1.items():
                if v != 1:
                    if k == "multiplier":
                        k = "x"
                    name += f"_{k}_{v}"
            if hn_2_name and hn_2_name != "None":
                name += f"_{re.sub(r' ', '_', merge_method)}"
                if merge_method == "Weighted Sum":
                    name += f"_{ws_ratio}"
                hn_2_name_sub = re.sub(r"\(.+\)", "", hn_2_name)
                name += f'_{hn_2_name_sub}'
                for k, v in opt_2.items():
                    if v != 1:
                        name += f"_{k}_{v}"
                if merge_method == "Add Difference":
                    hn_3_name_sub = re.sub(r"\(.+\)", "", hn_3_name)
                    name += f"_{hn_3_name_sub}_{ws_ratio}"
            if activation_func_strategy != "Keep Primary":
                name += f"_{sd_1['activation_func']}"
            if multiplier != 1:
                name += f"_x_{multiplier}"

        sd_1["name"] = name
        filename = os.path.join(folder_path, f'{name}.pt')
        torch.save(sd_1, filename)

        return_text += f"Hypernetwork saved to {filename}"
    if return_hn:
        return sd_1, return_text
    return return_text


def add_tab():
    def hypernetworks_with_none():
        return ["None"] + list(shared.hypernetworks.keys())

    def show_hn_info(hn_1, hn_2, hn_3):
        infotext = ""
        for hn in [hn_1, hn_2, hn_3]:
            if hn != "None":
                sd = torch.load(shared.hypernetworks[hn], map_location='cpu')
                infotext += f"{hn}:\n"
                infotext += f"Layers: {get_layers(sd)}\n"
                infotext += f"Structure: {get_layer_structure(sd)}\n"
                if "step" in sd:
                    infotext += f"Stepcount: {sd['step']}\n"
                if "activation_func" in sd:
                    infotext += f"Activation: {sd['activation_func']}\n"
                if "sd_checkpoint_name" in sd:
                    infotext += f"Checkpoint Name: {sd['sd_checkpoint_name']}\n"
                if "sd_checkpoint" in sd:
                    infotext += f"Checkpoint Hash: {sd['sd_checkpoint']}\n"
                if "weight_initiation" in sd:
                    infotext += f"Weight Initiation: {sd['weight_initiation']}\n"
                if "use_dropout" in sd:
                    infotext += f"Dropout: {sd['use_dropout']}\n"
                    if sd["use_dropout"]:
                        if "dropout_structure" in sd:
                            infotext += f"Dropout Structure: {sd['dropout_structure']}\n"

                infotext += "\n"

        return infotext

    with gr.Blocks(analytics_enabled=False) as modelmerger_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Accordion(label="Hypernetwork Modify Help", open=False):
                    helptext = gr.HTML(
                        """
                        <p><h1>Hi There! <img style="display: inline; vertical-align: text-bottom;" src="file=extensions/hypernetwork-modify/res/llama-up-and-down.gif"/></h1>
                        <br>This is my attempt at a hypernetwork modification tool. It's not perfect, but it should work for most cases.
                        <br>It's still in development, so if you find any bugs or have any suggestions, please let me know!
                        <br>
                        <br>To quickly get started you can use the default settings, and just select two hypernetworks to merge.
                        <br>You can use the Weighted sum slider to adjust the merge ratio, and the strength slider to adjust the strength of the merged hypernetwork.
                        <br>The resulting hypernetwork will be saved in the hypernetwork directory, in a folder called "~modified".
                        <br>You can also use this tool to modify a single hypernetwork, by not selecting a second hypernetwork.
                        <br>It will be named after the hypernetworks you merged, and the settings you used.
                        <br>
                        <br>The tool also comes with a modified X/Y Plot script to quickly test different settings.
                        <br><br>For more advanced usage you please refer to the help texts below.
                        """
                    )
                    with gr.Accordion(label="Hypernetwork Primer", open=False):
                        primer_text = gr.HTML(
                            """
                            <br>Hypernetworks are a type of neural network that adjusts the weights of another neural network.
                            <br>For our purposes you can think of their structure as a series of layers.
                            <br>The layer structure in most cases looks like this: 1,2,1 since it's the default.
                            <br>The first layer is always 1, and the last layer is always 1. so the middle layers are the ones that matter.
                            <br>The number of layers is often referred to as the width of the hypernetwork.
                            <br>Whereas the size of a layer is often referred to as the depth of the hypernetwork.
                            <br>So a wide network might look like this: 1,1.5,1.5,1.5,1 and a deep network might look like this: 1,4,1.
                            <br>
                            <br>Each Layer can have multiple sized modules, the defaults are 320, 640, 768 and 1280.
                            <br>The modules each control different aspects of the output image.
                            <br>In my experience it seems like the lower modules control the general composition of the image, 
                            <br>and the bigger ones control the finer details.
                            <br>Also generally the larger the module the more it will affect the output image.
                            <br>
                            <br>The way different networks are activated also varies.
                            <br>These so called "Activation Methods" dictate how the networks are activated.
                            <br>Most older networks use linear activation, but newer networks use all sorts of different activation methods.
                            <br>You can select the activation method you want to use in the "Activation Method" dropdown.
                            <br>For most cases you can just leave it on "Keep Primary".
                            <br>
                            <br>For more information on hypernetworks you should read the <b><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions?discussions_q=hypernetwork">Discussions</a></b> on the A1111 github.
                            """
                        )
                    with gr.Accordion(label="All Features", open=False):
                        features_text = gr.HTML(
                            """
                            <br>Here's some more info on the features of this tool.
                            <br>
                            <br>The Modification method radio menu lets you choose how the hypernetworks are merged.
                            <br>Weighted Sum is the default, and should be familiar from checkpoint merging.
                            <br>It works the same way, the networks add up to a strength of 1 and the slider lets you adjust the ratio.
                            <br>Add is a simple addition of the two networks, and the networks are not normalized.
                            <br>You should only use this if you are changing the strength of the networks.
                            <br>Subtract is a simple subtraction of the two networks, I'm not sure how useful this is, but it's there.
                            <br>Add Difference is the same as with checkpoint merging, it subtracts the tertiary from the secondary, and adds it to the primary.
                            <br>So a = a + (b - c) * ad slider.
                            <br>
                            <br>The Strength slider lets you adjust the strength of the merged hypernetwork, this uses the square root of the slider value.
                            <br>This is to approximate the strength slider in the hypernetwork settings.
                            <br>
                            <br>A problem i ran into during development is that hypernetworks can come with different layer structures, layer sizes and modules.
                            <br>For example one network might have a structure of 1,2,1 and another might have a structure of 1,1.5,1.5,1.5,1.
                            <br>This is where missing module strategy and resize strategy come in.
                            <br>
                            <br>Missing module strategy lets you choose what to do with modules that are missing in one of the networks.
                            <br>Add will copy the missing module from the other network.
                            <br>Remove will remove the missing module from both networks.
                            <br>
                            <br>Layer structure / Tensor resize strategy lets you choose what to do with layers and tensors that are mismatched.
                            <br>Extend will increase the size and number of the layer to match the largest network.
                            <br>Shrink will decrease the size and number of the layer to match the smallest network.
                            <br>
                            <br>Next up is the activation method, this lets you choose how the networks are activated.
                            <br>Keep Primary should be used in most cases, it will keep the activation method of the primary network.
                            <br>But feel free to experiment with the other options.
                            <br>
                            <br>The sliders let you adjust the strength of the networks and the strength of their modules.
                            <br>This can be useful if you want to see how different layers affect the output image.
                            <br>
                            <br>Finally the custom name option lets you choose a custom name for the merged hypernetwork.
                            <br>
                            <br>Thank you for reading. Hopefully this will be of some use to you! <img style="display: inline; vertical-align: text-bottom;" src="file=extensions/hypernetwork-modify/res/llama-eating-ramen.gif"/>
                            """
                        )
                            
                gr.HTML(value="<p>The modified hypernetwork will be saved in hypernetworks/~modified</p>")
                with gr.Row():
                    primary_model_name = gr.Dropdown(choices=hypernetworks_with_none(),
                                                     value="None",
                                                     elem_id="hn_modify_primary_model_name",
                                                     label="Primary Hypernetwork")
                    ui.create_refresh_button(primary_model_name, shared.reload_hypernetworks,
                                             lambda: {"choices": hypernetworks_with_none()},
                                             "hn_modify_refresh_1")
                    secondary_model_name = gr.Dropdown(choices=hypernetworks_with_none(),
                                                       value="None",
                                                       elem_id="hn_modify_secondary_model_name",
                                                       label="Secondary Hypernetwork")
                    ui.create_refresh_button(secondary_model_name, shared.reload_hypernetworks,
                                             lambda: {"choices": hypernetworks_with_none()},
                                             "hn_modify_refresh_2")

                    tertiary_model_name = gr.Dropdown(choices=hypernetworks_with_none(),
                                                      value="None",
                                                      elem_id="hn_modify_tertiary_model_name",
                                                      label="Tertiary Hypernetwork")
                    ui.create_refresh_button(tertiary_model_name, shared.reload_hypernetworks,
                                             lambda: {"choices": hypernetworks_with_none()},
                                             "hn_modify_refresh_3")

                strength = gr.Slider(minimum=-2.0, maximum=3.0, step=0.05,
                                     label='Sets the strength of resulting hypernetwork', value=1.0,
                                     elem_id="hn_modify_strength")

                ws_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                      label='Weighted Sum/Add Difference slider, only applies when ws/ad is selected',
                                      value=0.5,
                                      elem_id="hn_modify_strength")

                modify_method = gr.Radio(choices=["Weighted Sum", "Add", "Subtract", "Add Difference"],
                                         value="Weighted Sum",
                                         label="Modification Method", elem_id="hn_modify_interp_method")

                missing_module_strategy = gr.Radio(choices=["Add", "Ignore", "Remove"], value="Add",
                                                  label="Missing Module Strategy",
                                                  elem_id="hn_modify_missing_module_strategy")

                ls_tensor_resize_strategy = gr.Radio(choices=["Extend", "Shrink"], value="Extend",
                                                           label="Layer Structure / Tensor Resize Strategy",
                                                           elem_id="hn_modify_ls_tensor_resize_strategy")


                activation_func_strategy = gr.Dropdown(
                    choices=["Keep Primary", "Keep Secondary"] + hypernetworks.ui.keys,
                    value="Keep Primary",
                    label="Activation Function Strategy",
                    elem_id="hn_modify_activation_func_strategy")

                with gr.Row():
                    with gr.Column():
                        primary_multiplier = gr.Slider(minimum=-2.0, maximum=3.0, step=0.05,
                                                       label="Primary Strength", value=1.0,
                                                       elem_id="hn_modify_primary_strength")
                        primary_320 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                label="Primary 320 Layer", value=1.0,
                                                elem_id="hn_modify_primary_320")
                        primary_640 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                label="Primary 640 Layer", value=1.0,
                                                elem_id="hn_modify_primary_640")
                        primary_768 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                label="Primary 768 Layer", value=1.0,
                                                elem_id="hn_modify_primary_768")
                        primary_1024 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                 label="Primary 1024 Layer", value=1.0,
                                                 elem_id="hn_modify_primary_1024")
                        primary_1280 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                 label="Primary 1280 Layer", value=1.0,
                                                 elem_id="hn_modify_primary_1280")
                    with gr.Column():
                        secondary_multiplier = gr.Slider(minimum=-2.0, maximum=3.0, step=0.05,
                                                         label="Secondary Strength", value=1.0,
                                                         elem_id="hn_modify_secondary_strength")
                        secondary_320 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                  label="Secondary 320 Layer", value=1.0,
                                                  elem_id="hn_modify_secondary_320")
                        secondary_640 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                  label="Secondary 640 Layer", value=1.0,
                                                  elem_id="hn_modify_secondary_640")
                        secondary_768 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                  label="Secondary 768 Layer", value=1.0,
                                                  elem_id="hn_modify_secondary_768")
                        secondary_1024 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                   label="Secondary 1024 Layer", value=1.0,
                                                   elem_id="hn_modify_secondary_1024")
                        secondary_1280 = gr.Slider(minimum=-5.0, maximum=10.0, step=0.05,
                                                   label="Secondary 1280 Layer", value=1.0,
                                                   elem_id="hn_modify_secondary_1280")

                custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="hn_modify_custom_name")

                modelmerger_merge = gr.Button(elem_id="hn_modify_button", label="Modify", variant='primary')

            with gr.Column(variant='panel'):
                submit_result = gr.Textbox(elem_id="modelmerger_result", show_label=False)

            primary_model_name.change(fn=show_hn_info,
                                      inputs=[primary_model_name, secondary_model_name, tertiary_model_name],
                                      outputs=[submit_result])

            secondary_model_name.change(fn=show_hn_info,
                                        inputs=[primary_model_name, secondary_model_name, tertiary_model_name],
                                        outputs=[submit_result])

            tertiary_model_name.change(fn=show_hn_info,
                                       inputs=[primary_model_name, secondary_model_name, tertiary_model_name],
                                       outputs=[submit_result])

            modelmerger_merge.click(
                fn=modify_hypernetwork,
                inputs=[
                    primary_model_name,
                    secondary_model_name,
                    tertiary_model_name,
                    primary_multiplier, primary_320, primary_640, primary_768, primary_1024, primary_1280,
                    secondary_multiplier, secondary_320, secondary_640, secondary_768, secondary_1024, secondary_1280,
                    modify_method,
                    strength,
                    custom_name,
                    missing_module_strategy,
                    ls_tensor_resize_strategy,
                    activation_func_strategy,
                    ws_slider
                ],
                outputs=[
                    submit_result
                ])

    return [(modelmerger_interface, "Hypernetwork Modifier", "hn_modify")]


script_callbacks.on_ui_tabs(add_tab)
