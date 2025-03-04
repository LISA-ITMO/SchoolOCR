from PIL import Image
import io
import base64


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


image = Image.open("./cropped_tables/page_1.jpg")
base64_str = image_to_base64(image)
print(base64_str)
