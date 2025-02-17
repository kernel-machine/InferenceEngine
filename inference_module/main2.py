import zmq
import numpy as np
import cv2
import struct
import torch
import torchvision
from model_vivit import ModelVivit
import time

context = zmq.Context()
model = ModelVivit(hidden_layers=5)
auto_processing = model.get_image_processor()
model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 32 , 3, 224, 224).to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True, map_location=device))
model = model.module.eval()
model = torch.compile(model)
print("Compiling...")
model(dummy_input)

socket = context.socket(zmq.REQ)
socket.connect("tcp://172.19.0.3:5555")

def receive_images(data) -> tuple[list[cv2.Mat], bool]:
    """Estrae le immagini dal buffer binario ricevuto."""
    images = []
    offset = 0
    while offset < len(data):
        # Legge la dimensione dell'immagine
        img_size, is_infested = struct.unpack_from("I?", data, offset)
        offset += 5  # Sposta l'offset
        # Legge l'immagine
        img_bytes = struct.unpack_from(f"{img_size}s", data, offset)[0]
        offset += img_size  # Sposta l'offset
        # Decodifica e aggiunge alla lista
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        images.append(image)
    return images, is_infested

def pre_process_images(batch_data):
    processed_tensors = []
    for image in batch_data:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        processed_tensors.append(tensor)
    processed_tensors = torch.stack(processed_tensors)
    processed_tensors = processed_tensors.unsqueeze(0)
    processed_tensors = torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])(processed_tensors)
    return processed_tensors

batch = []
labels = []
for request in range(100):
    print("Sending request %s …" % request)
    socket.send(b"frames")
    batch_data = socket.recv()
    images,is_infested = receive_images(batch_data)
    images = pre_process_images(images).cuda()

    batch.append(images.squeeze(0))
    labels.append(is_infested)

    if len(batch)==4:
        batch = torch.stack(batch)
        print(f"Data shape: {batch.shape}")
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            prediction_logits = model(batch)
            torch.cuda.synchronize()
            end_time = time.time()
        inference_time = end_time - start_time
        predicted_classes = torch.sigmoid(prediction_logits).round().flatten().cpu()
        predicted_classes = list(map(lambda x:bool(x),predicted_classes))
        print(f"Recevied {len(images)} images -> {labels} | prediction: {predicted_classes} | Time: {(inference_time*1000)}ms")
        batch=[]
        labels = []
