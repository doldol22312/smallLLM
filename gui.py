import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import queue
import os
import sys
import json


def _enable_high_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        # Try per-monitor v2 first, then fall back.
        try:
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
            ctypes.windll.user32.SetProcessDpiAwarenessContext(
                DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            )
            return
        except Exception:
            pass

        try:
            # 2 = Per-monitor DPI aware, 1 = System DPI aware
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            return
        except Exception:
            pass

        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


def _configure_tk_scaling(root: tk.Tk) -> None:
    try:
        dpi = float(root.winfo_fpixels("1i"))
        root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass

class TinyLLMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TinyLLM Control Panel")
        self.root.geometry("1000x800")
        
        self.process = None
        self.queue = queue.Queue()
        self.is_running = False
        
        self.setup_ui()
        self.root.after(100, self.process_queue)
        
    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.train_frame = ttk.Frame(self.notebook)
        self.gen_frame = ttk.Frame(self.notebook)
        self.tools_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.train_frame, text="Training")
        self.notebook.add(self.gen_frame, text="Generation")
        self.notebook.add(self.tools_frame, text="Tools")
        
        self.setup_train_tab()
        self.setup_gen_tab()
        self.setup_tools_tab()
        
        # Log Area (Common)
        log_frame = ttk.LabelFrame(self.root, text="Logs")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=15, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bottom Controls
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill="x", padx=10, pady=5)
        
        self.stop_btn = ttk.Button(bottom_frame, text="Stop Current Process", command=self.stop_process, state="disabled")
        self.stop_btn.pack(side="right")
        
        self.clear_log_btn = ttk.Button(bottom_frame, text="Clear Logs", command=self.clear_logs)
        self.clear_log_btn.pack(side="right", padx=5)

    def make_scrollable(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_frame_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfigure(window_id, width=event.width)

        scroll_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def on_mousewheel(event):
            if sys.platform == "win32":
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                canvas.yview_scroll(int(-1 * event.delta), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        return scroll_frame

    def setup_train_tab(self):
        container = ttk.Frame(self.train_frame)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        scroll = self.make_scrollable(container)

        self.vars = {}

        io_group = ttk.LabelFrame(scroll, text="Data & Output")
        io_group.pack(fill="x", pady=5)
        self.add_file_picker(io_group, "Config (optional):", "config_path", "config/tiny_char_gpt.json", 0)
        self.add_file_picker(io_group, "Data Path:", "data_path", "data/tiny_corpus.txt", 1)
        self.add_dir_picker(io_group, "Output Dir:", "out_dir", "out/gui_run", 2)
        self.add_check(io_group, "Resume", "resume", False, 3)
        self.add_check(io_group, "Memmap dataset (tokens.bin)", "memmap_dataset", False, 4)
        self.add_option(io_group, "Device:", "device", ["auto", "cpu", "cuda"], 5)
        self.add_entry(io_group, "Seed:", "seed", "1337", 6)

        tok_group = ttk.LabelFrame(scroll, text="Tokenizer")
        tok_group.pack(fill="x", pady=5)
        self.add_option(tok_group, "Tokenizer:", "tokenizer", ["char", "bpe"], 0)
        self.add_entry(tok_group, "BPE vocab size:", "bpe_vocab_size", "2000", 1)
        self.add_entry(tok_group, "BPE min frequency:", "bpe_min_frequency", "2", 2)

        model_group = ttk.LabelFrame(scroll, text="Model")
        model_group.pack(fill="x", pady=5)
        self.add_entry(model_group, "Layers:", "n_layer", "4", 0)
        self.add_entry(model_group, "Heads:", "n_head", "4", 1)
        self.add_entry(model_group, "Embed dim:", "n_embd", "128", 2)
        self.add_entry(model_group, "Block size:", "block_size", "128", 3)
        self.add_entry(model_group, "Dropout:", "dropout", "0.1", 4)
        self.add_option(model_group, "Norm:", "norm", ["layernorm", "rmsnorm"], 5)
        self.add_entry(model_group, "Norm eps:", "norm_eps", "1e-5", 6)
        self.add_option(model_group, "MLP:", "mlp", ["gelu", "swiglu"], 7)
        self.add_option(model_group, "Pos encoding:", "pos_encoding", ["learned", "rope"], 8)
        self.add_entry(model_group, "RoPE base:", "rope_base", "10000.0", 9)
        self.add_entry(model_group, "Sliding window (0=off):", "sliding_window", "0", 10)
        self.add_check(model_group, "QK-Norm", "qk_norm", False, 11)
        self.add_entry(model_group, "QK-Norm eps:", "qk_norm_eps", "1e-5", 12)
        self.add_check(model_group, "Parallel block", "parallel_block", False, 13)
        self.add_entry(model_group, "LayerDrop (0..1):", "layerdrop", "0.0", 14)
        self.add_check(model_group, "Scaled init", "scaled_init", True, 15)
        self.add_check(model_group, "Weight tying", "weight_tying", False, 16)

        train_group = ttk.LabelFrame(scroll, text="Training")
        train_group.pack(fill="x", pady=5)
        self.add_entry(train_group, "Max steps:", "max_steps", "2000", 0)
        self.add_entry(train_group, "Batch size:", "batch_size", "64", 1)
        self.add_entry(train_group, "Grad accum steps:", "grad_accum_steps", "1", 2)
        self.add_entry(train_group, "Learning rate:", "learning_rate", "3e-4", 3)
        self.add_option(train_group, "LR schedule:", "lr_schedule", ["constant", "cosine"], 4)
        self.add_entry(train_group, "LR warmup steps:", "lr_warmup_steps", "200", 5)
        self.add_entry(train_group, "LR min:", "lr_min", "3e-5", 6)
        self.add_option(train_group, "Optimizer:", "optimizer", ["adamw", "lion"], 7)
        self.add_entry(train_group, "Lion beta1:", "lion_beta1", "0.9", 8)
        self.add_entry(train_group, "Lion beta2:", "lion_beta2", "0.99", 9)
        self.add_entry(train_group, "Weight decay:", "weight_decay", "0.1", 10)
        self.add_entry(train_group, "Grad clip:", "grad_clip", "1.0", 11)
        self.add_check(train_group, "Grad checkpointing", "grad_checkpointing", False, 12)
        self.add_entry(train_group, "Label smoothing:", "label_smoothing", "0.0", 13)
        self.add_entry(train_group, "Z-loss:", "z_loss", "0.0", 14)

        eval_group = ttk.LabelFrame(scroll, text="Eval & Sampling")
        eval_group.pack(fill="x", pady=5)
        self.add_entry(eval_group, "Eval interval (0=off):", "eval_interval", "200", 0)
        self.add_entry(eval_group, "Eval iters:", "eval_iters", "50", 1)
        self.add_entry(eval_group, "Log interval:", "log_interval", "50", 2)
        self.add_entry(eval_group, "Sample interval (0=off):", "sample_interval", "200", 3)
        self.add_entry(eval_group, "Sample prompt:", "train_prompt", "", 4)
        self.add_entry(eval_group, "Sample tokens:", "train_sample_chars", "400", 5)
        self.add_entry(eval_group, "Sample temperature:", "train_temperature", "1.0", 6)
        self.add_entry(eval_group, "Sample top-k:", "train_top_k", "0", 7)
        self.add_entry(eval_group, "Sample top-p:", "train_top_p", "1.0", 8)
        self.add_entry(eval_group, "Sample repetition penalty:", "train_repetition_penalty", "1.0", 9)
        self.add_check(eval_group, "Sample with KV cache", "train_kv_cache", False, 10)

        perf_group = ttk.LabelFrame(scroll, text="Performance")
        perf_group.pack(fill="x", pady=5)
        self.add_check(perf_group, "AMP", "amp", True, 0)
        self.add_check(perf_group, "torch.compile", "compile", False, 1)
        self.add_option(perf_group, "Compile backend:", "compile_backend", ["inductor", "aot_eager", "eager"], 2)
        self.add_option(perf_group, "Compile mode:", "compile_mode", ["default", "reduce-overhead", "max-autotune"], 3)
        self.add_check(perf_group, "Compile fullgraph", "compile_fullgraph", False, 4)
        self.add_option(perf_group, "Compile dynamic:", "compile_dynamic", ["auto", "true", "false"], 5)

        dist_group = ttk.LabelFrame(scroll, text="Distributed (DDP)")
        dist_group.pack(fill="x", pady=5)
        self.add_check(dist_group, "Enable DDP (torchrun)", "ddp", False, 0)
        self.add_entry(dist_group, "Processes / GPUs:", "ddp_nproc", "1", 1)
        self.add_option(dist_group, "DDP backend:", "ddp_backend", ["auto", "nccl", "gloo"], 2)
        self.add_check(dist_group, "find_unused_parameters", "ddp_find_unused_parameters", False, 3)
        self.add_entry(dist_group, "DDP timeout (s):", "ddp_timeout", "1800", 4)

        btn_train = ttk.Button(scroll, text="Start Training", command=self.start_training)
        btn_train.pack(fill="x", pady=10)

    def setup_gen_tab(self):
        container = ttk.Frame(self.gen_frame)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Generation Settings
        gen_group = ttk.LabelFrame(container, text="Inference Settings")
        gen_group.pack(fill="x", pady=5)
        
        self.add_dir_picker(gen_group, "Model Dir (out_dir):", "gen_out_dir", "out/tiny_char_gpt", 0)
        self.add_entry(gen_group, "Prompt:", "prompt", "Once upon a time", 1)
        self.add_entry(gen_group, "Tokens to generate:", "sample_chars", "200", 2)
        self.add_entry(gen_group, "Temperature:", "temperature", "1.0", 3)
        self.add_entry(gen_group, "Top K (0=off):", "top_k", "0", 4)
        self.add_entry(gen_group, "Top P:", "top_p", "1.0", 5)
        self.add_entry(gen_group, "Repetition penalty:", "repetition_penalty", "1.0", 6)
        self.add_check(gen_group, "Use KV Cache", "kv_cache", True, 7)
        
        btn_gen = ttk.Button(container, text="Generate Text", command=self.start_generation)
        btn_gen.pack(fill="x", pady=10)

    def setup_tools_tab(self):
        container = ttk.Frame(self.tools_frame)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        
        wiki_group = ttk.LabelFrame(container, text="Clean WikiText")
        wiki_group.pack(fill="x", pady=5)
        
        self.add_file_picker(wiki_group, "Input File:", "wiki_input", "data/wiki.train.TXT", 0)
        self.add_file_picker(wiki_group, "Output File:", "wiki_output", "data/wiki.clean.txt", 1)
        self.add_check(wiki_group, "Keep Headings", "wiki_keep_headings", False, 2)
        
        btn_clean = ttk.Button(wiki_group, text="Clean WikiText", command=self.start_cleaning)
        btn_clean.grid(row=3, column=0, columnspan=2, pady=10)

    def start_cleaning(self):
        if self.is_running:
            return
        
        cmd = [sys.executable, "prepare_wikitext.py"]
        cmd.extend(["--input", self.vars["wiki_input"].get()])
        cmd.extend(["--output", self.vars["wiki_output"].get()])
        if self.vars["wiki_keep_headings"].get():
            cmd.append("--keep_headings")
            
        self.run_command(cmd)

    def add_entry(self, parent, label, var_name, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        var = tk.StringVar(value=default)
        self.vars[var_name] = var
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        parent.columnconfigure(1, weight=1)

    def add_check(self, parent, label, var_name, default, row):
        var = tk.BooleanVar(value=default)
        self.vars[var_name] = var
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)

    def add_option(self, parent, label, var_name, options, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        var = tk.StringVar(value=options[0])
        self.vars[var_name] = var
        ttk.Combobox(parent, textvariable=var, values=options, state="readonly").grid(row=row, column=1, sticky="ew", padx=5, pady=2)

    def add_file_picker(self, parent, label, var_name, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        var = tk.StringVar(value=default)
        self.vars[var_name] = var
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew")
        ttk.Entry(frame, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(frame, text="...", width=3, command=lambda: self.pick_file(var)).pack(side="right")

    def add_dir_picker(self, parent, label, var_name, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        var = tk.StringVar(value=default)
        self.vars[var_name] = var
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew")
        ttk.Entry(frame, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(frame, text="...", width=3, command=lambda: self.pick_dir(var)).pack(side="right")

    def pick_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def pick_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def log(self, message):
        self.queue.put(message)

    def clear_logs(self):
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
                self.log_text.configure(state='disabled')
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def start_training(self):
        if self.is_running:
            return

        ddp_enabled = False
        try:
            ddp_enabled = self.vars["ddp"].get() and int(self.vars["ddp_nproc"].get()) > 1
        except Exception:
            ddp_enabled = False

        if ddp_enabled:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(int(self.vars["ddp_nproc"].get())),
                "train_tinyllm.py",
                "--ddp",
            ]
        else:
            cmd = [sys.executable, "train_tinyllm.py"]

        config_path = self.vars.get("config_path")
        if config_path and config_path.get().strip():
            cmd.extend(["--config", config_path.get().strip()])
        
        # Add arguments
        cmd.extend(["--data_path", self.vars["data_path"].get()])
        cmd.extend(["--out_dir", self.vars["out_dir"].get()])
        if self.vars["resume"].get(): cmd.append("--resume")
        if self.vars["memmap_dataset"].get(): cmd.append("--memmap_dataset")
        else: cmd.append("--no-memmap_dataset")

        cmd.extend(["--device", self.vars["device"].get()])
        seed = self.vars["seed"].get().strip()
        if seed:
            cmd.extend(["--seed", seed])

        if ddp_enabled:
            cmd.extend(["--ddp_backend", self.vars["ddp_backend"].get()])
            if self.vars["ddp_find_unused_parameters"].get():
                cmd.append("--ddp_find_unused_parameters")
            else:
                cmd.append("--no-ddp_find_unused_parameters")
            cmd.extend(["--ddp_timeout", self.vars["ddp_timeout"].get()])

        cmd.extend(["--tokenizer", self.vars["tokenizer"].get()])
        if self.vars["tokenizer"].get() == "bpe":
            cmd.extend(["--bpe_vocab_size", self.vars["bpe_vocab_size"].get()])
            cmd.extend(["--bpe_min_frequency", self.vars["bpe_min_frequency"].get()])

        cmd.extend(["--n_layer", self.vars["n_layer"].get()])
        cmd.extend(["--n_head", self.vars["n_head"].get()])
        cmd.extend(["--n_embd", self.vars["n_embd"].get()])
        cmd.extend(["--block_size", self.vars["block_size"].get()])
        cmd.extend(["--dropout", self.vars["dropout"].get()])
        cmd.extend(["--norm", self.vars["norm"].get()])
        cmd.extend(["--norm_eps", self.vars["norm_eps"].get()])
        cmd.extend(["--mlp", self.vars["mlp"].get()])
        cmd.extend(["--pos_encoding", self.vars["pos_encoding"].get()])
        cmd.extend(["--rope_base", self.vars["rope_base"].get()])
        cmd.extend(["--sliding_window", self.vars["sliding_window"].get()])

        if self.vars["qk_norm"].get(): cmd.append("--qk_norm")
        else: cmd.append("--no-qk_norm")
        cmd.extend(["--qk_norm_eps", self.vars["qk_norm_eps"].get()])

        if self.vars["parallel_block"].get(): cmd.append("--parallel_block")
        else: cmd.append("--no-parallel_block")
        cmd.extend(["--layerdrop", self.vars["layerdrop"].get()])

        if self.vars["scaled_init"].get(): cmd.append("--scaled_init")
        else: cmd.append("--no-scaled_init")

        if self.vars["weight_tying"].get(): cmd.append("--weight_tying")
        else: cmd.append("--no-weight_tying")

        cmd.extend(["--max_steps", self.vars["max_steps"].get()])
        cmd.extend(["--batch_size", self.vars["batch_size"].get()])
        cmd.extend(["--grad_accum_steps", self.vars["grad_accum_steps"].get()])
        cmd.extend(["--learning_rate", self.vars["learning_rate"].get()])
        cmd.extend(["--lr_schedule", self.vars["lr_schedule"].get()])
        cmd.extend(["--lr_warmup_steps", self.vars["lr_warmup_steps"].get()])
        cmd.extend(["--lr_min", self.vars["lr_min"].get()])
        cmd.extend(["--optimizer", self.vars["optimizer"].get()])
        if self.vars["optimizer"].get() == "lion":
            cmd.extend(["--lion_beta1", self.vars["lion_beta1"].get()])
            cmd.extend(["--lion_beta2", self.vars["lion_beta2"].get()])
        cmd.extend(["--weight_decay", self.vars["weight_decay"].get()])
        cmd.extend(["--grad_clip", self.vars["grad_clip"].get()])
        if self.vars["grad_checkpointing"].get(): cmd.append("--grad_checkpointing")
        else: cmd.append("--no-grad_checkpointing")
        cmd.extend(["--label_smoothing", self.vars["label_smoothing"].get()])
        cmd.extend(["--z_loss", self.vars["z_loss"].get()])

        cmd.extend(["--eval_interval", self.vars["eval_interval"].get()])
        cmd.extend(["--eval_iters", self.vars["eval_iters"].get()])
        cmd.extend(["--log_interval", self.vars["log_interval"].get()])
        cmd.extend(["--sample_interval", self.vars["sample_interval"].get()])
        cmd.extend(["--prompt", self.vars["train_prompt"].get()])
        cmd.extend(["--sample_chars", self.vars["train_sample_chars"].get()])
        cmd.extend(["--temperature", self.vars["train_temperature"].get()])
        cmd.extend(["--top_k", self.vars["train_top_k"].get()])
        cmd.extend(["--top_p", self.vars["train_top_p"].get()])
        cmd.extend(["--repetition_penalty", self.vars["train_repetition_penalty"].get()])
        if self.vars["train_kv_cache"].get(): cmd.append("--kv_cache")
        else: cmd.append("--no-kv_cache")

        if self.vars["amp"].get(): cmd.append("--amp")
        else: cmd.append("--no-amp")

        if self.vars["compile"].get(): cmd.append("--compile")
        else: cmd.append("--no-compile")
        cmd.extend(["--compile_backend", self.vars["compile_backend"].get()])
        cmd.extend(["--compile_mode", self.vars["compile_mode"].get()])
        if self.vars["compile_fullgraph"].get(): cmd.append("--compile_fullgraph")
        else: cmd.append("--no-compile_fullgraph")

        compile_dynamic = self.vars["compile_dynamic"].get()
        if compile_dynamic == "true":
            cmd.append("--compile_dynamic")
        elif compile_dynamic == "false":
            cmd.append("--no-compile_dynamic")

        self.run_command(cmd)

    def start_generation(self):
        if self.is_running:
            return
        
        cmd = [sys.executable, "train_tinyllm.py", "--generate_only"]
        cmd.extend(["--out_dir", self.vars["gen_out_dir"].get()])
        cmd.extend(["--prompt", self.vars["prompt"].get()])
        cmd.extend(["--sample_chars", self.vars["sample_chars"].get()])
        cmd.extend(["--temperature", self.vars["temperature"].get()])
        cmd.extend(["--top_k", self.vars["top_k"].get()])
        cmd.extend(["--top_p", self.vars["top_p"].get()])
        cmd.extend(["--repetition_penalty", self.vars["repetition_penalty"].get()])
        
        if self.vars["kv_cache"].get(): cmd.append("--kv_cache")
        else: cmd.append("--no-kv_cache")
        
        self.run_command(cmd)

    def run_command(self, cmd):
        self.is_running = True
        self.stop_btn.config(state="normal")
        self.log(f"Running: {' '.join(cmd)}\n\n")
        
        def target():
            try:
                # Use creationflags to prevent terminal window popup on Windows if desired
                # but for now we want to see it or just pipe it.
                self.process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    universal_newlines=True
                )
                
                for line in self.process.stdout:
                    self.log(line)
                
                self.process.wait()
                self.log(f"\nProcess finished with exit code {self.process.returncode}\n")
            except Exception as e:
                self.log(f"\nError: {str(e)}\n")
            finally:
                self.is_running = False
                self.stop_btn.config(state="disabled")
                self.process = None

        threading.Thread(target=target, daemon=True).start()

    def stop_process(self):
        if self.process:
            if messagebox.askyesno("Confirm", "Are you sure you want to stop the current process?"):
                # On Windows, taskkill might be cleaner for subprocess trees
                if sys.platform == "win32":
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(self.process.pid)])
                else:
                    self.process.terminate()
                self.log("\nProcess stopping...\n")

if __name__ == "__main__":
    _enable_high_dpi_awareness()
    root = tk.Tk()
    _configure_tk_scaling(root)
    app = TinyLLMGUI(root)
    root.mainloop()
