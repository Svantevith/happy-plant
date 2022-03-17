import re
import requests
import os
from urllib.request import Request, urlopen
import io
import bs4
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import NoReturn
from pytesseract import pytesseract

pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def show_image(image: np.ndarray, window_name: str) -> NoReturn:
    # Display image
    cv2.imshow(window_name, image)
    # Keep the window open until user presses any key
    cv2.waitKey(0)
    # Destroy present screen windows
    cv2.destroyAllWindows()


def scrape_image(url: str) -> np.ndarray:
    agent = {'User-Agent': 'Mozilla/5.0'}
    page = requests.get(url, headers=agent)
    bs = bs4.BeautifulSoup(page.content, 'html.parser')
    img_src = bs.find('img', {'alt': 'plant-health-final'}).attrs['src']
    img_req = Request(img_src, headers=agent)
    image_file = io.BytesIO(urlopen(img_req).read())
    return Image.open(image_file)


def get_text_from_image(img_buffer: str) -> list:
    if os.path.isfile(img_buffer):
        img = Image.open(img_buffer)
    else:
        img = scrape_image(img_buffer)

    # Crop the image
    cropped_img = img.crop((0, img.size[1] * 0.03, img.size[0], img.size[1] * 0.97))

    # Convert the image to gray scale
    img_gray = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary form using thresholding
    # Binary representation is required for coloured images.
    # Binarization makes it easier for Tesseract to detect text correctly
    img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Display the image
    # show_image(img_threshold, 'Binarized image')

    # Feed the image to tesseract
    details = pytesseract.image_to_data(
        image=img_threshold,
        output_type=pytesseract.Output.DICT,
        # config=r'--oem 3 --psm 6',  # Configure parameters for tesseract
        lang='eng'
    )

    total_boxes = len(details['text'])

    boxed_img = img_threshold.copy()
    for i in range(total_boxes):
        # Confidence should be generally between 30-40%
        if int(details['conf'][i]) >= 30:
            x, y, w, h = (
                details['left'][i],
                details['top'][i],
                details['width'][i],
                details['height'][i]
            )
            boxed_img = cv2.rectangle(boxed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display image with boxes
    # show_image(boxed_img, "Image with text detection")

    parsed_lines = []
    words = []
    last_word = ''
    for word in details['text']:
        if re.search('\\w+', word):
            words.append(word)
            last_word = word
        if (words and last_word and not word) or (word == details['text'][-1]):
            parsed_lines.append(words)
            words = []

    return [' '.join(line) for line in parsed_lines if line]


def get_data_from_image(img_buffer: str) -> pd.DataFrame:
    labels = [
        'Bacterial Spot',
        'Bacterial Blight',
        'Ralstonia solanacearum',
        'Thielaviopsis',
        'Aphids',
        'Cucumber Mosaic Virus',
        'Botrytis',
        'Downy Mildew',
        'Cylindrocladium',
        'Angular Leaf Spot',
        'Rhizoctonia',
        'Spider Mites',
        'Anthracnose',
        'Mealybugs',
    ]
    headers = ['Disease', 'Reason', 'Symptoms', 'Treatment']
    lines = get_text_from_image(img_buffer)
    data = {k: [] for k in headers}
    for i, label in enumerate(labels):
        idx_prev = lines.index(label)
        try:
            idx_next = lines.index(labels[i + 1])
        except IndexError:
            idx_next = len(lines)
        disease = [line for line in lines[idx_prev:idx_next]]
        fields = [-1] + [i for i, line in enumerate(disease) if line.startswith('WHAT')] + [len(disease)]
        for k, field in enumerate(fields[:-1]):
            data[headers[k]].append(' '.join(disease[field + 1: fields[k + 1]]))
    return pd.DataFrame.from_dict(data)
