"""  Parse data from cian.ru
https://github.com/lenarsaitov/cianparser
"""

import os
import cianparser
import datetime
import pandas as pd

moscow_parser = cianparser.CianParser(location="Москва")


def main():
    """
    Function docstring
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path_dir = os.path.join(file_dir, "..", "data", "raw")
    if not os.path.exists(csv_path_dir):
        os.makedirs(csv_path_dir, exist_ok=True)

    n_rooms = 1
    while n_rooms <= 4:
        t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        csv_path = os.path.join(csv_path_dir, f'{n_rooms}_{t}.csv')
        
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 10,
                "object_type": "secondary"
            })
        df = pd.DataFrame(data)

        df.to_csv(csv_path,
                  encoding='utf-8',
                  index=False)
        n_rooms +=1

if __name__ == '__main__':
    main()