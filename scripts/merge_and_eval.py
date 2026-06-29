#!/usr/bin/env python3
"""
acent LoRA merge + eval pipeline
"""
import subprocess
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run(cmd, desc):
    print(f"\n=== {desc} ===", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: {desc} (rc={result.returncode})", flush=True)
    return result.returncode

# Step 1: Export (merge LoRA weights into base model)
rc1 = run(
    "python -m paddleformers.cli.cli export "
    "--model_name_or_path /home/xufei/acent/output "
    "--output_dir /home/xufei/acent/export",
    "Export LoRA weights"
)

if rc1 != 0 or not os.path.exists("/home/xufei/acent/export/model_state.pdparams"):
    print("CLI export failed or output missing model_state.pdparams, trying merge_config approach...", flush=True)
    rc1 = run(
        "python -c \""
        "import sys; sys.path.insert(0, '/home/xufei/PaddleFormers-develop'); "
        "from paddleformers.merge import merge_main; "
        "merge_main('/home/xufei/acent/merge_config.json')"
        "\"",
        "Fallback merge via merge_config.json"
    )

# Check if export succeeded
export_dir = "/home/xufei/acent/export"
model_files = ["model_state.pdparams"]
if not any(os.path.exists(os.path.join(export_dir, f)) for f in model_files):
    # Check for safetensors
    has_safetensors = any("model-" in f and f.endswith(".safetensors") 
                          for f in os.listdir(export_dir) if os.path.isdir(export_dir))
    if not has_safetensors:
        print("ERROR: No model weights found in export directory!", flush=True)
        print("Files in export dir:", os.listdir(export_dir) if os.path.isdir(export_dir) else "NOT A DIR", flush=True)
        sys.exit(1)

# Step 2: Evaluate
rc2 = run(
    "python /home/xufei/tibetan_ocr_lora/full_eval.py "
    "--model_path /home/xufei/acent/export/ "
    "--data_path /home/xufei/acent/acent_paddle_test.jsonl "
    "--image_dir / "
    "--output_path /home/xufei/acent/eval_acent_results.json "
    "--log_every 10 "
    "--max_new_tokens 512",
    "Evaluate on acent test set"
)

print(f"\n=== Pipeline complete (export_rc={rc1}, eval_rc={rc2}) ===", flush=True)
