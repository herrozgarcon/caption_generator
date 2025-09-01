import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image, ImageEnhance, ImageOps
import random

# Load models
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_text_model():
    return pipeline("text-generation", model="gpt2", max_length=60, num_return_sequences=1)

processor, model = load_caption_model()
stylizer = load_text_model()

# Emojis + hashtags
emoji_map = {
    "food": ["🍕", "🍔", "🍩", "🍫"],
    "travel": ["🌍", "✈️", "🌅", "🏞️"],
    "animal": ["🐾", "🐶", "🐱", "🦁"],
    "aesthetic": ["✨", "🌸", "💫", "🌈"],
    "funny": ["😂", "🤣", "😜"],
    "romantic": ["💕", "🌹", "😍"],
    "motivational": ["🔥", "💯", "🚀"]
}
hashtags_map = {
    "food": ["#Foodie", "#Yum", "#InstaFood"],
    "travel": ["#Wanderlust", "#TravelGram", "#Adventure"],
    "animal": ["#DogLife", "#CatVibes", "#Petstagram"],
    "aesthetic": ["#Dreamy", "#InstaAesthetic", "#GoodVibesOnly"],
    "funny": ["#LOL", "#GoodTimes", "#MemeVibes"],
    "romantic": ["#LoveVibes", "#Soulmate", "#CoupleGoals"],
    "motivational": ["#KeepPushing", "#StayStrong", "#Mindset"]
}

# Apply filters
def apply_filter(img, filter_type):
    if filter_type == "Original":
        return img
    elif filter_type == "Black & White":
        return ImageOps.grayscale(img)
    elif filter_type == "Sepia":
        sepia = ImageOps.colorize(ImageOps.grayscale(img), "#704214", "#C0A080")
        return sepia
    elif filter_type == "Bright Boost":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.5)
    elif filter_type == "Cool Tone":
        r, g, b = img.split()
        b = b.point(lambda i: i * 1.3)
        return Image.merge("RGB", (r, g, b))
    return img

# UI
st.title("📸 AI Instagram Caption + Filter Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
style = st.selectbox("Choose style", ["aesthetic", "funny", "romantic", "motivational", "travel", "food", "animal"])
length = st.radio("Caption length", ["Short", "Medium", "Long"])
filter_choice = st.selectbox("Choose a photo filter", ["Original", "Black & White", "Sepia", "Bright Boost", "Cool Tone"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    filtered_image = apply_filter(image, filter_choice)
    st.image(filtered_image, caption=f"Preview with {filter_choice}", use_column_width=True)

    if st.button("Generate Captions"):
        # Step 1: Raw caption
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs, max_length=20, num_beams=3)
        raw_caption = processor.decode(out[0], skip_special_tokens=True).capitalize()

        # Step 2: Stylize captions
        st.subheader("✨ Captions:")
        all_captions = []
        for i in range(3):
            prompt = f"Rewrite this as a {style} Instagram caption ({length} length): {raw_caption}"
            styled = stylizer(prompt)[0]['generated_text']
            styled = styled.replace(prompt, "").strip()

            # Add emojis + hashtags
            decorated = f"{styled} {' '.join(random.sample(emoji_map.get(style, []), 2))}\n{' '.join(random.sample(hashtags_map.get(style, []), 2))}"

            all_captions.append(decorated)
            st.write(f"**Option {i+1}:** {decorated}")

            # Copy-to-clipboard
            st.markdown(
                f"""<button onclick="navigator.clipboard.writeText('{decorated.replace("'", "\\'")}')">
                📋 Copy Option {i+1}
                </button>""",
                unsafe_allow_html=True,
            )

        # Save all captions as text
        captions_txt = "\n\n".join(all_captions)
        st.download_button("⬇ Download All Captions", captions_txt, "captions.txt")

        # Save filtered image option
        st.download_button("⬇ Download Edited Image", filtered_image.tobytes(), "filtered_image.png")
