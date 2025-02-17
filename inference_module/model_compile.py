import argparse
import torch
import torch_tensorrt
from model_vivit import ModelVivit

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, required=True, help="Path to weights")
args.add_argument("--hidden_layers", type=int, default=0, help="Path to weights")

args = args.parse_args()
model = ModelVivit(hidden_layers=args.hidden_layers)
auto_processing = model.get_image_processor()
model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 32 , 3, 224, 224).to(device)
model.load_state_dict(torch.load(args.model, weights_only=True, map_location=device))


model = model.module.eval()
standard_out = model(dummy_input)
trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=[dummy_input], enabled_precisions={torch.int8}, torch_executed_ops={"aten::add", "aten::mul"})  # Esegui solo alcune operazioni con TensorRT)
torch_tensorrt.save(trt_gm, "trt.ep", inputs=[dummy_input])
print(f"Model exported")
