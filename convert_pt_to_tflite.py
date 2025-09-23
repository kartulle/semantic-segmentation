import argparse, os, subprocess, sys
import torch, onnx
from onnxsim import simplify

def export_onnx(pt_path, size, onnx_out="unet_pet.onnx"):
    ts = torch.jit.load(pt_path, map_location="cpu").eval()
    dummy = torch.zeros(1, 3, size, size)
    # 1ª tentativa: exportador clássico
    try:
        torch.onnx.export(
            ts, dummy, onnx_out,
            input_names=["input"], output_names=["logit"],
            opset_version=13, do_constant_folding=True, dynamic_axes=None
        )
        return onnx_out
    except Exception as e1:
        # 2ª tentativa: exportador novo (dynamo)
        try:
            from torch.onnx import dynamo_export
            export_out = dynamo_export(ts, dummy)
            export_out.save(onnx_out)
            return onnx_out
        except Exception as e2:
            raise RuntimeError(f"Falha ao exportar ONNX.\nClássico: {e1}\nDynamo: {e2}")

def simplify_onnx(onnx_in, onnx_out="unet_pet_simp.onnx"):
    m = onnx.load(onnx_in)
    m_simp, ok = simplify(m)
    if not ok:
        raise RuntimeError("Falha ao simplificar ONNX")
    onnx.save(m_simp, onnx_out)
    return onnx_out

def onnx_to_tf(onnx_path, tf_dir="unet_tf"):
    import sys, subprocess
    # sem --output_nhwc; usa o mesmo Python da venv
    cmd = [sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", tf_dir]
    subprocess.check_call(cmd)
    return tf_dir

def tf_to_tflite(tf_saved_dir="unet_tf/saved_model", tflite_path="unet_pet_fp32.tflite"):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_dir)
    # Sem quantização => FP32
    tflite = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite)
    return tflite_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="caminho do TorchScript .pt")
    ap.add_argument("--size", type=int, default=256, help="input H=W (ex.: 256)")
    args = ap.parse_args()

    print("==> Exportando ONNX…")
    onnx_raw = export_onnx(args.pt, args.size)
    print("ONNX bruto:", onnx_raw)

    print("==> Simplificando ONNX…")
    onnx_simp = simplify_onnx(onnx_raw)
    print("ONNX simplificado:", onnx_simp)

    print("==> Convertendo ONNX -> TF (NHWC)…")
    tf_dir = onnx_to_tf(onnx_simp)
    print("SavedModel:", os.path.join(tf_dir, "saved_model"))

    print("==> Convertendo TF -> TFLite FP32…")
    tflite_path = tf_to_tflite(os.path.join(tf_dir, "saved_model"))
    print("OK:", tflite_path, "| tamanho (MB):", os.path.getsize(tflite_path)/1e6)

    # Checagem rápida de I/O
    import tensorflow as tf, numpy as np
    interp = tf.lite.Interpreter(model_path=tflite_path); interp.allocate_tensors()
    inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
    print("INPUT :", inp)   # esperado: [1, size, size, 3] float32
    print("OUTPUT:", out)   # esperado: [1, size, size, 1] float32 (logits)
    x = np.zeros((1, args.size, args.size, 3), np.float32)
    interp.set_tensor(inp["index"], x); interp.invoke()
    y = interp.get_tensor(out["index"])
    print("Sanity:", y.shape, "min/max:", float(y.min()), float(y.max()))

if __name__ == "__main__":
    sys.exit(main())
