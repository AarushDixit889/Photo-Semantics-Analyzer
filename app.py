import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
def get_semantics(image):
    print("GETTING SEMANTICS")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    raw_image = Image.open(image).convert('RGB')
    inputs=processor(raw_image,"",return_tensors="pt")
    out = model.generate(**inputs)
    out = list(processor.decode(out[0], skip_special_tokens=True))
    out[0] = out[0].upper()
    out = "".join(out)
    return out

st.set_page_config(page_title="Photo Semantics Analyzer",layout="wide")

st.title("Photo Semantics Analyzer")

file=st.file_uploader("Put your file")
if file is not None:
    st.sidebar.image(file)
    answer=get_semantics(file)
    st.write(answer)
