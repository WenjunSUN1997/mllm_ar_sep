from transformers import Blip2Model, Blip2Processor
import requests
from PIL import Image
from model_config.blip2_classification import Blip2ModelForTokenClassification

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda:0")

model = Blip2Model.from_pretrained('Salesforce/blip2-opt-2.7b')


# model = Blip2ModelForTokenClassification.from_pretrained('Salesforce/blip2-opt-2.7b',
#                                                          text_model_name='bert-base-uncased',
#                                                          ignore_mismatched_sizes=True)
model.to('cuda:0')
out = model(**inputs)
print()