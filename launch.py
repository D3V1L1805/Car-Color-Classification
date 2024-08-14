import gradio as gr
from PIL import Image
from ultralytics import YOLO
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)

model = YOLO("Car_Colours_Classify_v1.pt")

def detect_objects(images):
    results = model(images)
    classes = {0: "beige", 1: "black", 2: "blue", 3: "brown", 4: "gold", 5: "green", 6: "grey", 7: "orange", 8: "pink", 9: "purple", 10: "red", 11: "silver", 12: "tan", 13: "white", 14: "yellow"}
    names = []
    for result in results:
        probs = result.probs.top1
        names.append(classes[probs])
    return names

def create_solutions(image_urls, names, file_ids):
    solutions = []
    for image_url, prediction, file_id in zip(image_urls, names, file_ids):
        obj = {"url": image_url, "answer": [prediction], "qcUser" : None, "normalfileID" : file_id}
        solutions.append(obj)
    return solutions

# def send_results_to_api(solutions, url):
#     headers = {"Content-Type": "application/json"}
#     try:
#         logging.info(f"Sending results to API at {url} with data: {solutions}")
#         # response = requests.patch(url, json = {"solutions":solutions})  # Set a timeout   headers=headers,, timeout=60
#         data = {"solutions":solutions}
#         response = requests.patch(url, data=json.dumps(data), headers=headers)
#         response.raise_for_status()
#         logging.info(f"Response from API: {response.text}")
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Failed to send results to API: {e}")
#         return {"error": f"Failed to send results to API: {str(e)}"}

def process_images(params):
    try:
        params = json.loads(params)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON input: {e.msg} at line {e.lineno} column {e.colno}")
        return {"error": f"Invalid JSON input: {e.msg} at line {e.lineno} column {e.colno}"}

    image_urls = params.get("urls", [])
    if not params.get("normalfileID",[]):
        file_ids = [None]*len(image_urls)
    else:
        file_ids = params.get("normalfileID",[])
    # api = params.get("api", "")
    # job_id = params.get("job_id", "")

    if not image_urls:
        logging.error("Missing required parameters: 'urls'")
        return {"error": "Missing required parameters: 'urls'"}

    try:
        images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
    except Exception as e:
        logging.error(f"Error loading images: {e}")
        return {"error": f"Error loading images: {str(e)}"}

    names = detect_objects(images)
    solutions = create_solutions(image_urls, names, file_ids)

    # result_url = f"{api}/{job_id}"
    # response = send_results_to_api(solutions, result_url)

    return json.dumps({"solutions": solutions})

inputt = gr.Textbox(label="Parameters (JSON format) Eg. {'urls':['a.jpg','b.jpg']}")
outputs = gr.JSON()

application = gr.Interface(fn=process_images, inputs=inputt, outputs=outputs, title="Car Colour Classification with API Integration")
application.launch()