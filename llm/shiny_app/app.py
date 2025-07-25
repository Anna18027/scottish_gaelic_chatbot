from shiny import App, reactive, render, ui
import os
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


BASE_MODEL_DIR = os.path.join("llm", "trained_models")
VARYING_PARAMS = ["subset_size"]

def get_base_models():
    return sorted([
        name for name in os.listdir(BASE_MODEL_DIR)
        if os.path.isdir(os.path.join(BASE_MODEL_DIR, name))
    ])

def get_run_types(base_model):
    model_path = os.path.join(BASE_MODEL_DIR, base_model)
    if not os.path.isdir(model_path):
        return []
    return sorted([
        name for name in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, name))
    ])

def read_grid_params_lines(base_model, run_type):
    file_path = os.path.join(BASE_MODEL_DIR, base_model, run_type, "grid_params.txt")
    if not os.path.isfile(file_path):
        return []
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def parse_line_to_params(line):
    tokens = line.split()
    params = {}
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("--"):
            param = tokens[i][2:]
            if i + 1 < len(tokens):
                value = tokens[i+1]
                params[param] = value
                i += 2
            else:
                i += 1
        else:
            i += 1
    return params

def gather_all_params_and_values():
    all_params = set()
    values_dict = {}
    for base_model in get_base_models():
        values_dict[base_model] = {}
        for run_type in get_run_types(base_model):
            values_dict[base_model][run_type] = {}
            lines = read_grid_params_lines(base_model, run_type)
            for line in lines:
                params = parse_line_to_params(line)
                for p, v in params.items():
                    all_params.add(p)
                    values_dict[base_model][run_type].setdefault(p, set()).add(v)
    return sorted(all_params), values_dict

ALL_PARAMS, PARAM_VALUES = gather_all_params_and_values()

def params_match(metadata, selected_params):
    for k, v in selected_params.items():
        meta_val = metadata.get(k)
        if meta_val is None:
            return False
        if isinstance(meta_val, list):
            meta_val_str = ",".join(map(str, meta_val))
        else:
            meta_val_str = str(meta_val)
        if str(v) != meta_val_str:
            return False
    return True

param_selectors_varying = [
    ui.input_select(f"param_{param}", f"Select {param}:", choices=[""])
    for param in ALL_PARAMS if param in VARYING_PARAMS
]

param_selectors_static = [
    ui.input_select(f"param_{param}", f"Select {param}:", choices=[""])
    for param in ALL_PARAMS if param not in VARYING_PARAMS
]

app_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel(
            "Explore Model Results",
            ui.h2("Model Selection"),
            ui.row(
                ui.column(
                    4,
                    ui.input_select("base_model", "Select a base model:", choices=[""] + get_base_models()),
                    ui.input_select("run_type", "Select a run type:", choices=[""]),
                    ui.h3("Varying Params"),
                    *param_selectors_varying,
                    ui.h3("Static Params"),
                    *param_selectors_static,
                ),
                ui.column(
                    8,
                    ui.tags.div(
                        ui.output_text("selected_path"),
                        ui.output_text_verbatim("grid_params", placeholder=True),
                        ui.output_text_verbatim("metadata_contents", placeholder=True),
                        ui.output_text("matched_log_folder"),
                        ui.output_image("loss_plot"),
                        style="width: 100%; height: 100%; display: block; margin-top: 0; padding-top: 0;"
                    )
                )
            )
        ),
        ui.nav_panel(
            "Generate",
            ui.h3("Model selection"),
            ui.row(
                ui.column(4, ui.input_select("gen_base_model", "Select a base model:", choices=[""] + get_base_models())),
                ui.column(4, ui.input_select("gen_run_type", "Select a run type:", choices=[""])),
                ui.column(4, ui.input_select("gen_subset_size", "Select a subset size:", choices=[""]))
            ),
            ui.tags.hr(),
            ui.h3("Text generation"),
            ui.output_text("generate_selection_text"),
            ui.output_text("gen_matched_log_folder"),
            ui.input_text("gen_prompt", "Input your prompt here:", placeholder=""),
            ui.input_action_button("gen_submit_btn", "Generate"),
            ui.output_text("gen_prompt_display"),
            ui.output_text("gen_completion")
        )
    ),
    style="margin-top: 0; padding-top: 0;"
)





def server(input, output, session):
    @reactive.Effect
    def update_run_type_choices():
        base_model = input.base_model()
        if base_model:
            run_types = get_run_types(base_model)
            ui.update_select("run_type", choices=[""] + run_types)
        else:
            ui.update_select("run_type", choices=[""])

    @reactive.Effect
    def update_param_choices():
        base_model = input.base_model()
        run_type = input.run_type()
        if base_model and run_type and base_model in PARAM_VALUES and run_type in PARAM_VALUES[base_model]:
            for param in ALL_PARAMS:
                vals = sorted(PARAM_VALUES[base_model][run_type].get(param, []))
                if not vals:
                    ui.update_select(f"param_{param}", choices=[""], selected=None, label=f"Select {param}:")
                    continue
                label = f"Select {param}:"
                if len(vals) > 1:
                    label += " *"
                ui.update_select(f"param_{param}", choices=vals, selected=vals[0], label=label)
        else:
            for param in ALL_PARAMS:
                ui.update_select(f"param_{param}", choices=[""], selected=None, label=f"Select {param}:")

    @output
    @render.text
    def selected_path():
        if input.base_model() and input.run_type():
            return f"Selected path: trained_models/{input.base_model()}/{input.run_type()}"
        return "No model/run selected"

    @output
    @render.text
    def grid_params():
        base_model = input.base_model()
        run_type = input.run_type()
        if not base_model or not run_type:
            return "[Select both a base model and a run type above]"
        try:
            file_path = os.path.join(BASE_MODEL_DIR, base_model, run_type, "grid_params.txt")
            if not os.path.isfile(file_path):
                return "[grid_params.txt not found]"
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"[Error reading grid_params.txt: {e}]"

    @output
    @render.text
    def metadata_contents():
        base_model = input.base_model()
        run_type = input.run_type()
        if not base_model or not run_type:
            return "[Select both a base model and a run type]"
        selected_params = {}
        for param in ALL_PARAMS:
            val = input[f"param_{param}"]()
            if val:
                selected_params[param] = val
        logs_dir = os.path.join(BASE_MODEL_DIR, base_model, run_type)
        if not os.path.isdir(logs_dir):
            return "[Run directory not found]"
        log_folders = [f for f in os.listdir(logs_dir) if f.startswith("logs_") and os.path.isdir(os.path.join(logs_dir, f))]
        if not log_folders:
            return "[No logs_x folder found]"
        for lf in log_folders:
            meta_path = os.path.join(logs_dir, lf, "metadata.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    if params_match(meta, selected_params):
                        return json.dumps(meta, indent=4)
                except Exception as e:
                    return f"[Error reading metadata.json: {e}]"
        return "[No matching metadata.json found]"

    @output
    @render.text
    def matched_log_folder():
        base_model = input.base_model()
        run_type = input.run_type()
        if not base_model or not run_type:
            return "[Select both a base model and a run type]"
        selected_varying_params = {
            param: input[f"param_{param}"]()
            for param in VARYING_PARAMS
            if input[f"param_{param}"]()
        }
        lines = read_grid_params_lines(base_model, run_type)
        if not lines:
            return "[No grid_params.txt or it's empty]"
        for idx, line in enumerate(lines):
            parsed = parse_line_to_params(line)
            parsed_str = {k: str(v) for k, v in parsed.items()}
            selected_str = {k: str(v) for k, v in selected_varying_params.items()}
            if all(selected_str.get(k) == parsed_str.get(k) for k in selected_str):
                log_folder = f"logs_{idx+1}"
                full_path = os.path.join(BASE_MODEL_DIR, base_model, run_type, log_folder)
                if os.path.isdir(full_path):
                    return f"Matched folder: {log_folder} (line {idx+1})"
                else:
                    return f"[Match found in grid_params.txt line {idx+1}, but {log_folder} does not exist]"
        return "[No matching line in grid_params.txt found]"

    @output
    @render.image
    def loss_plot():
        base_model = input.base_model()
        run_type = input.run_type()
        if not base_model or not run_type:
            return None
        selected_varying_params = {
            param: input[f"param_{param}"]()
            for param in VARYING_PARAMS
            if input[f"param_{param}"]()
        }
        lines = read_grid_params_lines(base_model, run_type)
        for idx, line in enumerate(lines):
            parsed = parse_line_to_params(line)
            parsed_str = {k: str(v) for k, v in parsed.items()}
            selected_str = {k: str(v) for k, v in selected_varying_params.items()}
            if all(selected_str.get(k) == parsed_str.get(k) for k in selected_str):
                log_folder = f"logs_{idx+1}"
                image_path = os.path.join(BASE_MODEL_DIR, base_model, run_type, log_folder, "loss_curve.png")
                if os.path.isfile(image_path):
                    return {"src": image_path} 
        return None
    
    @reactive.Effect
    def update_gen_run_type_choices():
        base_model = input.gen_base_model()
        if base_model:
            run_types = get_run_types(base_model)
            ui.update_select("gen_run_type", choices=[""] + run_types)
        else:
            ui.update_select("gen_run_type", choices=[""])
        # Also clear subset size when base model changes
        ui.update_select("gen_subset_size", choices=[""])

    @reactive.Effect
    def update_gen_subset_size_choices():
        base_model = input.gen_base_model()
        run_type = input.gen_run_type()
        if base_model and run_type:
            # Extract subset_size values from PARAM_VALUES if available
            subset_sizes = []
            if base_model in PARAM_VALUES and run_type in PARAM_VALUES[base_model]:
                subset_sizes = sorted(PARAM_VALUES[base_model][run_type].get("subset_size", []))
            ui.update_select("gen_subset_size", choices=[""] + subset_sizes)
        else:
            ui.update_select("gen_subset_size", choices=[""])

    @output
    @render.text
    def generate_selection_text():
        bm = input.gen_base_model() or "<none>"
        rt = input.gen_run_type() or "<none>"
        ss = input.gen_subset_size() or "<none>"
        if bm == "<none>" or rt == "<none>" or ss == "<none>":
            return "Please select base model, run type, and subset size."
        return f"You are using {bm} / {rt} / {ss}"

    @reactive.Calc
    def gen_prompt_triggered():
        input.gen_submit_btn()  # depend on button click
        with reactive.isolate():
            return input.gen_prompt()

    @output
    @render.text
    def gen_prompt_display():
        prompt = gen_prompt_triggered()
        return f"Your prompt: {prompt}" if prompt else "No prompt entered yet."

    @output
    @render.text
    def gen_matched_log_folder():
        base_model = input.gen_base_model()
        run_type = input.gen_run_type()
        subset_size = input.gen_subset_size()
        if not base_model or not run_type or not subset_size:
            return "[Select base model, run type, and subset size]"

        selected_params = {"subset_size": subset_size}
        lines = read_grid_params_lines(base_model, run_type)
        if not lines:
            return "[No grid_params.txt or it's empty]"

        for idx, line in enumerate(lines):
            parsed = parse_line_to_params(line)
            parsed_str = {k: str(v) for k, v in parsed.items()}
            selected_str = {k: str(v) for k, v in selected_params.items()}
            if all(selected_str.get(k) == parsed_str.get(k) for k in selected_str):
                log_folder = f"logs_{idx+1}"
                full_path = os.path.join(BASE_MODEL_DIR, base_model, run_type, log_folder)
                if os.path.isdir(full_path):
                    return f"Matched folder: {log_folder} (line {idx+1})"
                else:
                    return f"[Match found in grid_params.txt line {idx+1}, but {log_folder} does not exist]"
        return "[No matching line in grid_params.txt found]"

    GENERATION_PARAMS = {
        "max_new_tokens": 30,
        "temperature": 0.8,
        "top_p": 0.95,
        "do_sample": True,
    }

    DEVICE = "cpu"

    def load_model_and_tokenizer(model_path, device=DEVICE):
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        return model, tokenizer

    def generate_completion(prompt, model, tokenizer, generation_params, device=DEVICE):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_params
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded[len(prompt):].strip()


    @output
    @render.text
    def gen_completion():
        prompt = gen_prompt_triggered()
        base_model = input.gen_base_model()
        run_type = input.gen_run_type()
        subset_size = input.gen_subset_size()

        if not prompt or not base_model or not run_type or not subset_size:
            return "[Please enter a prompt and select base model/run type/subset size]"

        lines = read_grid_params_lines(base_model, run_type)
        if not lines:
            return "[grid_params.txt not found]"

        for idx, line in enumerate(lines):
            parsed = parse_line_to_params(line)
            if str(parsed.get("subset_size")) == subset_size:
                model_path = os.path.join(BASE_MODEL_DIR, base_model, run_type, f"logs_{idx+1}", "saved_model")
                if not os.path.isdir(model_path):
                    return f"[Model folder not found: {model_path}]"
                try:
                    model, tokenizer = load_model_and_tokenizer(model_path)
                    completion = generate_completion(prompt, model, tokenizer, GENERATION_PARAMS)
                    return f"Model Completion:\n{completion}"
                except Exception as e:
                    return f"[Error loading model or generating: {e}]"

        return "[No matching model found for selected subset size]"



app = App(app_ui, server)
