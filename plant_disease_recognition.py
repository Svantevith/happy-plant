import os
import re
from datetime import datetime
from typing import NoReturn, Union
from dotenv import load_dotenv

import sqlite3
import numpy as np
import pandas as pd
from PIL import UnidentifiedImageError

from neural_networks.imagenet import IHappyPlant
from helper_scripts.export_excel import save_excel
from scraping.image_scraping.image_scraper import get_data_from_image
from scraping.web_scraping.web_scraper import get_data_from_web

"""
Plant Diseases, Garden Pests, and Possible Treatments [web]
@ https://www.fondation-louisbonduelle.org/

14 Common Plant Diseases and How to Treat Them [images]
@ https://www.proflowers.com/
"""

# Load environmental variables
load_dotenv(r'D:\PyCharm Professional\Projects\HappyPlant\.env')
DATABASE_PATH = os.getenv('DATABASE_PATH')
EXPORT_DIR = os.getenv('EXPORT_DIR')
FONDATION_URL, PRO_FLOWERS_URL = (os.getenv(k) for k in ['FONDATION_URL', 'PRO_FLOWERS_URL'])


def find_words(line: str) -> list:
    words = re.findall('[A-Za-z\\-]+', line)
    return [line, *words] if len(words) > 1 else words


def format_text(text: Union[str, list], indent: int = 1, enum: bool = False) -> str:
    if isinstance(text, str):
        text = text.split('\n')
    tab = '\t' * indent
    if enum:
        return ''.join([f'\n{tab} [{i}] {word}' for i, word in enumerate(text, start=1)])
    return ''.join([f'\n{tab} - {word}' for word in text])


def extract_keywords(text: str) -> Union[list, np.ndarray]:
    if re.search('\\w+', text):
        keywords = re.split('\\s?,\\s?', text)
        if len(keywords) == 1:
            return find_words(keywords[0])
        return np.ravel([find_words(keyword) for keyword in keywords])
    return []


def get_output_path():
    if not os.path.isdir(EXPORT_DIR):
        os.mkdir(EXPORT_DIR)
    file_name = f"happy_plant_disease_queries_{datetime.strftime(datetime.now(), '%Y%m%d%H%M')}.xlsx"
    return os.path.join(EXPORT_DIR, file_name)


def get_matches(df: pd.DataFrame, keywords: list) -> pd.Series:
    matches = {}
    for keyword in keywords:
        for idx in df.index:
            disease = df.loc[idx, 'Disease']
            if re.search(f'{keyword}', disease, re.IGNORECASE):
                return pd.Series({disease: 1}, dtype=int)
            for col in df.columns:
                found = len(re.findall(keyword, df.loc[idx, col], re.IGNORECASE))
                if found > 0:
                    matches[disease] = matches.get(disease, 1) + found
    return pd.Series(matches, dtype=int)


def concat_cells(x: str) -> str:
    lines = re.split('\\.\\s', ' '.join(x))
    return '.\n'.join(lines) if len(lines) > 1 else lines[0]


def get_database(load_from_database: bool = True, save: bool = False) -> pd.DataFrame:
    with sqlite3.connect(DATABASE_PATH) as connection:
        if load_from_database:
            # return pd.read_excel(DATABASE_PATH)
            return pd.read_sql(
                sql="SELECT * FROM plant_diseases",
                con=connection
            )
        df_from_web = get_data_from_web(FONDATION_URL)
        df_from_image = get_data_from_image(PRO_FLOWERS_URL)
        database = pd.concat(
            objs=[df_from_image, df_from_web]
        ).groupby('Disease').agg(lambda x: concat_cells(x)).reset_index()
        if save:
            database.to_sql(
                name='plant_diseases',
                con=connection,
                index=False,
                if_exists='replace'
            )
        return database


def disease_description(query: str, is_predicted: bool = False) -> NoReturn:
    keywords = extract_keywords(query)
    if len(keywords) == 0:
        print("[🌺] You haven't provided any valid keywords. We hope your plant is right as rain!")
        return

    data = get_database(load_from_database=True)
    matches = get_matches(data, keywords)
    if not matches.empty:
        if is_predicted:
            diseases = [matches.idxmax(), ]
        else:
            diseases = matches[matches == matches.max()].index
            found = diseases.shape[0]
            print(f"[🌷] {found} suitable match{'es' if found > 1 else ''} found based on extracted keywords:")
            print(f'{format_text(keywords, indent=1)}')
        for i, disease in enumerate(diseases, start=1):
            row = data.loc[data['Disease'] == disease].reset_index(inplace=False)
            print(
                f"\n\t[💮 {'Prediction' if is_predicted else i}] "
                + f'Potential disease, accompanying symptoms, possible reasons and appropriate treatment hints:\n'
                + f'\n\t\t[🦠] The plant disease indicated by the provided keywords is most likely:\n'
                + f'\t\t{format_text(disease, indent=2)}.\n'
                + '\n\t\t[🌡️] The symptoms and damage to the plant may be as following:\n'
                + f"\t\t{format_text(row.loc[0, 'Symptoms'], indent=2)}\n"
                + '\n\t\t[🔬] Possible cause of appearance of these diseases or pests is favorably:\n'
                + f"\t\t{format_text(row.loc[0, 'Reason'], indent=2)}\n"
                + '\n\t\t[🌸] The appropriate treatment involves:\n'
                + f"\t\t{format_text(row.loc[0, 'Treatment'], indent=2)}"
            )

        if re.match('y', input("\n[🦕] Would you like to save your session? y/[n]: "), re.IGNORECASE):
            queries = data[data['Disease'].isin(diseases)]
            output_path = get_output_path()
            save_excel(queries, output_path)

        return

    if is_predicted:
        print(f"[💧] Unfortunately we haven't found any information concerning {query} in our database.")

    else:
        print(
            f"[💧] Unfortunately we haven't found any suitable matches for provided keywords:"
            f"{format_text(keywords, indent=1)}"
        )

    print(
        f"\n[📩] We highly apologize for any inconveniences and shortcomings of available database."
        f"[📊] Furthermore we kindly encourage our users to report encountered errors to HappyPlant support!"
    )


def plant_disease_recognition() -> NoReturn:
    features = ['Search Engine', 'Image Net']
    output_message = "[🥑] Thank you for trusting us - HappyPlant is keeping track on your plant's health conditions 💚"
    print(f"\n[🧠] iHappyPlant - intelligent plant disease recognition & treatment:")
    print(format_text(features, indent=1, enum=True))
    try:
        choice = int(input('\n[💦] Choose feature: '))
        print(f'[🚀] Launching {features[choice - 1]}...\n')
        if choice == 1:
            query = input('[🌱] Please provide comma-separated keywords describing the visual defects of your plant: ')
            disease_description(query)
        elif choice == 2:
            i_happy_plant = IHappyPlant()
            user_input = input('[🌱] Please provide path to the image presenting pathological leaf: ')
            print("[⏳] Droplets 'plinking' sound is our iHappyPlant making predictions ...")
            y_pred = i_happy_plant.predict_disease(user_input)
            if re.match('Healthy', y_pred, re.IGNORECASE):
                print(f"\n\t[🍒] It seems like your plant is fit as a fiddle!")
                return
            elif re.match('Background', y_pred, re.IGNORECASE):
                print(f"\n\t[🐻] Unfortunately, iHappyPlant could not identify any plant on the uploaded image.")
                return
            else:
                disease_description(y_pred, is_predicted=True)
    except (ValueError, IndexError, UnidentifiedImageError, OSError) as err:
        print(
            f'\n\n[❌] Wrong input provided: {err}\n'
            f'[🍁] We hope your plant is safe and sound!'
        )
    finally:
        print(f"\n\n{output_message}")


if __name__ == '__main__':
    plant_disease_recognition()
