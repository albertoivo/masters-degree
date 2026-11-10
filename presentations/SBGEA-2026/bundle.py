#!/usr/bin/env python3
import base64
import os
import re
import sys

def detect_mime(file_path):
    with open(file_path, "rb") as f:
        head = f.read(16)
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    elif head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    elif head.startswith(b"RIFF") and b"WEBP" in head:
        return "image/webp"
    elif file_path.endswith('.svg'):
        return "image/svg+xml"
    return "application/octet-stream"

def get_base64_data_uri(file_path):
    mime_type = detect_mime(file_path)
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{data}"

def bundle(input_path, output_path):
    base_dir = os.path.dirname(os.path.abspath(input_path))
    
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 1. Encode all 28 sequence frames for the All-Sky animation
    seq_dir = os.path.join(base_dir, "assets", "seq")
    seq_frames = []
    for i in range(28):
        fname = f"f{i:02d}.jpg"
        fpath = os.path.join(seq_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing sequence frame: {fpath}")
        seq_frames.append(get_base64_data_uri(fpath))
    
    # Replace the JS dynamic sequence loader (line 1859 in index.html)
    target_js_pattern = r"const N = 28, frames = Array\.from\(\{ length: N \}, \(_, i\) => \{ const im = new Image\(\); im\.src = `assets/seq/f\$\{String\(i\)\.padStart\(2, '0'\)\}\.jpg`; return im \}\);"
    
    # Format sequence frames as JS array
    seq_js = "const seqFrames = [\n      " + ",\n      ".join(f'"{uri}"' for uri in seq_frames) + "\n    ];\n    const N = 28, frames = seqFrames.map(src => { const im = new Image(); im.src = src; return im });"
    
    if not re.search(target_js_pattern, content):
        raise ValueError("Could not find dynamic all-sky sequence loader pattern in JS!")
    
    content = re.sub(target_js_pattern, seq_js, content, count=1)
    
    # 2. Encode and replace all static image src attributes
    cache = {}
    
    def replace_src(match):
        rel_path = match.group(1)
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Referenced asset not found: {full_path}")
        if rel_path not in cache:
            cache[rel_path] = get_base64_data_uri(full_path)
        return f'src="{cache[rel_path]}"'
    
    content = re.sub(r'src="(assets/[^"]+)"', replace_src, content)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Bundled successfully:")
    print(f"  Input:  {input_path} ({os.path.getsize(input_path):,} bytes)")
    print(f"  Output: {output_path} ({os.path.getsize(output_path):,} bytes)")
    print(f"  Embedded {len(cache)} unique static assets and {len(seq_frames)} animation frames.")

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(dir_path, "index.html")
    output_file = os.path.join(dir_path, "index-standalone.html")
    bundle(input_file, output_file)
