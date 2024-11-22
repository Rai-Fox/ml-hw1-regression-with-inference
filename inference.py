import pandas as pd
import numpy as np
import random
import re
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

random.seed(42)
np.random.seed(42)

class DataPreprocessor:

    def preprocess_data(self, df):
        df = df.copy()
        df = self.preprocess_data_with_units(df)
        df = self.preprocess_torque(df)
        df = self.fill_na_medians(df)
        df = self.cast_fields(df)
        df = self.process_name(df)
        df = self.process_categorical_features(df)
        df = self.add_new_features(df)
        return df

    def preprocess_torque(self, df):
        def split_torque(torque):
            if pd.isna(torque):
                return np.nan, np.nan
            
            torque = torque.lower().replace(',', '').replace(' kgm at', 'kgm@').replace('nm at', 'nm@').replace('/', '').replace('  ', ' ').strip()
            G = 9.81
            float_number = '(\d+(\.\d+)?)'
            min_max_rpm = f'(({float_number}[-~])?{float_number})'
            # print(torque)


            pattern = f'{float_number} ?(nm)?@? {min_max_rpm} ?(rpm)?'
            match = re.fullmatch(pattern, torque)
            if match:
                # print('   ', match.groups())
                torque = float(match.group(1))
                rpm = float(match.group(8))
                # print('   ', torque, rpm)
                return torque, rpm
            
            pattern = f'{float_number}kgm@ {min_max_rpm} ?rpm'
            match = re.fullmatch(pattern, torque)
            if match:
                # print('   ', match.groups())
                torque = float(match.group(1)) * G
                rpm = float(match.group(7))
                # print('   ', torque, rpm)
                return torque, rpm
            
            pattern = f'{float_number}@ {min_max_rpm}\(kgm@ rpm\)'
            match = re.fullmatch(pattern, torque)
            if match:
                # print('   ', match.groups())
                torque = float(match.group(1)) * G
                rpm = float(match.group(7))
                # print('   ', torque, rpm)
                return torque, rpm
            
            pattern = f'{float_number}nm'
            match = re.fullmatch(pattern, torque)
            if match:
                # print('   ', match.groups())
                torque = float(match.group(1))
                rpm = np.nan
                # print('   ', torque, rpm)
                return torque, rpm
            
            # 380nm(38.7kgm)@ 2500rpm
            pattern = f'{float_number}nm\({float_number}kgm\)@ {min_max_rpm} ?rpm'
            match = re.fullmatch(pattern, torque)
            if match:
                # print('   ', match.groups())
                torque = float(match.group(1))
                rpm = float(match.group(9))
                # print('   ', torque, rpm)
                return torque, rpm
            
            # 51nm@ 4000+-500rpm
            pattern = f'{float_number}nm@ {float_number}\+-{float_number} ?rpm'
            match = re.fullmatch(pattern, torque)
            if match:
                # print('   ', match.groups())
                torque = float(match.group(1))
                rpm = float(match.group(3)) + float(match.group(5))
                # print('   ', torque, rpm)
                return torque, rpm
            
            # 48@ 3000+-500(nm@ rpm)
            pattern = f'{float_number}@ {float_number}\+-{float_number}\(nm@ rpm\)'
            match = re.fullmatch(pattern, torque)
            if match:
                # print('   ', match.groups())
                torque = float(match.group(1))
                rpm = float(match.group(3)) + float(match.group(5))
                # print('   ', torque, rpm)
                return torque, rpm
            
            # 110(11.2)@ 4800
            pattern = f'{float_number}\({float_number}\)@ {min_max_rpm}'
            match = re.fullmatch(pattern, torque)
            if match:
                # print('   ', match.groups())
                torque = float(match.group(1))
                rpm = float(match.group(9))
                # print('   ', torque, rpm)
                return torque, rpm
            
            raise ValueError(f'Unknown pattern: {torque}')
        
        df = df.copy()
        df['torque'], df['max_torque_rpm'] = zip(*df['torque'].apply(split_torque))
        return df


    def preprocess_data_with_units(self, df):
        df = df.copy()
        df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
        df['max_power'] = df['max_power'].str.replace(' bhp', '').apply(lambda x: float(x) if x else np.nan)
        df['engine'] = df['engine'].str.replace(' CC', '').astype(float)

        return df


    def fill_na_medians(self, df):
        medians = pickle.load(open('medians.pkl', 'rb'))

        for column in df.isna().sum()[df.isna().sum() > 0].index:
            df[column] = df[column].fillna(medians[column])
        return df


    def cast_fields(self, df):
        df = df.copy()

        df['seats'] = df['seats'].astype(int)
        df['engine'] = df['engine'].astype(int)

        return df


    def process_name(self, df):
        def transform_name(value):
            value = ' '.join(value.split()[0:2])
            return value

        df = df.copy()

        # Оставим только первые два слова в названии
        df['name'] = df['name'].apply(transform_name)
        return df


    def process_categorical_features(self, df):
        df = df.copy()
        cat_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
        num_columns = ['mileage', 'km_driven', 'max_power', 'year', 'engine', 'max_torque_rpm', 'torque']

        encoder = pickle.load(open('encoder.pkl', 'rb'))

        X_encoded = encoder.transform(df[cat_columns])

        X_encoded = pd.concat([pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out()), df[num_columns]], axis=1)
        return X_encoded


    def add_new_features(self, df):
        df = df.copy()
        df['mileage_engine'] = df['mileage'] * df['engine']
        df['mileage_max_power'] = df['mileage'] * df['max_power']
        df['mileage_torque'] = df['mileage'] * df['torque']
        df['engine_max_power'] = df['engine'] * df['max_power']
        df['engine_torque'] = df['engine'] * df['torque']
        df['max_power_torque'] = df['max_power'] * df['torque']

        df["year_2"] = df["year"] ** 2

        return df
    
class ModelInference:
        def __init__(self):
            self.model = pickle.load(open('model.pkl', 'rb'))
    
        def predict(self, data):
            return self.model.predict(data)
        
app = FastAPI()
model = ModelInference()
preprocessor = DataPreprocessor()

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    preprocessed_df = preprocessor.preprocess_data(df)
    prediction = model.predict(preprocessed_df)
    return prediction[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = pd.DataFrame([item.dict() for item in items])
    preprocessed_df = preprocessor.preprocess_data(df)
    prediction = model.predict(preprocessed_df)
    return list(prediction)
