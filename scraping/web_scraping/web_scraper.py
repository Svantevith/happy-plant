import requests
import bs4
import pandas as pd
import re


def get_cells(_table: bs4.Tag) -> list:
    return [cell.text for cell in _table.find_all('span', 'vc_table_content')]


def get_table(_bs: bs4.BeautifulSoup) -> bs4.Tag:
    return _bs.find('table', 'vc-table-plugin-theme-classic_green')


def get_beautiful_soup(url):
    page = requests.get(url)
    return bs4.BeautifulSoup(page.content, 'html.parser')


def clean_cells(_cells: list):
    return list(map(lambda s: s.strip('\n'), _cells))


def scrape_web(url: str) -> pd.DataFrame:
    # Scrape necessary table cells using BeautifulSoup
    bs = get_beautiful_soup(url)
    disease_table = get_table(bs)
    table_cells = get_cells(disease_table)

    # Prepare data for the DataFrame
    for i in range(0, len(table_cells), 4):
        row = clean_cells(table_cells[i: i + 4])
        if all(len(cell) > 0 for cell in row):
            yield row


def get_data_from_web(url: str) -> pd.DataFrame:
    df = pd.DataFrame(
        data=[row for row in scrape_web(url)],
        columns=['Disease', 'Symptoms', 'Reason', 'Treatment']
    )
    df['Disease'] = df['Disease'].apply(lambda s: re.sub('\\s?or\\s?', '', s).title())
    return df
