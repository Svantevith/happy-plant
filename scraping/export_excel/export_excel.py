import tkinter as tk
import tkinter.font as tk_font
from math import trunc
from typing import NoReturn

import pandas as pd
import xlsxwriter
from PIL import ImageFont
from _tkinter import TclError


def save_excel(df: pd.DataFrame, filename: str) -> NoReturn:
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet('Sheet1')

    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({
        'bold': True,
        'border': 1,
        'font_name': 'Calibri',
        'font_size': 11,
        'align': 'center',
        'bg_color': '#339933'

    })

    cell_format = {
        'bg_color': '',
        'border': 1,
        'font_name': 'Calibri',
        'font_size': 11
    }

    # Write some data headers.
    for i, letter in enumerate(range(ord('A'), ord('A') + df.shape[1])):
        worksheet.write(f'{chr(letter)}1', df.columns[i], bold)

    # Iterate over the data and write it out row by row.
    for i, idx in enumerate(df.index):
        for j, col in enumerate(df.columns):
            if col == 'Disease':
                cell_format['bg_color'] = '#b3e6b3'
            else:
                cell_format['bg_color'] = '#ecf9ec'

            # Start from the first cell below the headers.
            worksheet.write(i + 1, j, df.loc[idx, col], workbook.add_format(cell_format))

    # Autofit column width
    for i, col in enumerate(df.columns):
        longest = col if len(col) > len(max(df[col], key=len)) else max(df[col], key=len)
        width = pixel_width(longest)
        worksheet.set_column(i, i, width)

    # Output the Excel file
    workbook.close()
    print(f'[💾] Session has been successfully exported as {filename}')


def pixel_width(text: str, font_family='Calibri', font_size=11, padding=2.5) -> float:
    try:
        tk.Frame().destroy()
        reference_font = tk_font.Font(family=font_family, size=font_size)
        width = reference_font.measure(text)
        return trunc(width / 7) + padding
    except TclError:
        try:
            font = ImageFont.truetype(f'{font_family.lower()}.ttf', font_size)
            width = font.getsize(text)[0]
            return width * 0.2 + padding
        except OSError:
            return len(text) + 2 * padding
