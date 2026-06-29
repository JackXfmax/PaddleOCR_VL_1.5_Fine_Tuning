import os

charset_file = "/home/xufei/tibet_acent/tibetan_charset_clean.txt"
with open(charset_file, "r", encoding="utf-8") as f:
    chars = [line.strip() for line in f if line.strip()]
s = "".join(chars)

print("charset length:", len(s))

yaml_content = "# @package _global_\nmodel:\n  charset_train: " + repr(s) + "\n"

out_path = "/home/xufei/parseq/configs/charset/tibetan.yaml"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("Written to:", out_path)
print("First few chars:", repr(s[:10]))
