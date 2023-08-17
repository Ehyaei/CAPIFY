from folktables import ACSDataSource, ACSIncome
import logging
import os
import pandas as pd


def ASC_fetcher(states, survey_year, horizon, survey, file_name):
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey)
    data = data_source.get_data(states=states, download=True)
    income, labels, _ = ACSIncome.df_to_pandas(data)
    income['label'] = labels
    features = ['SEX', 'RAC1P', 'AGEP', 'POBP', 'MAR', 'SCHL', 'OCCP', 'WKHP', 'COW']
    labels = 'label'
    columns = features + [labels]
    ASC = income.loc[:, columns]
    print(income.head())
    ASC.to_csv(file_name, index=False)


def datasets_fetcher():
    logging.basicConfig(level=logging.INFO)

    # Fetch ASCIncome data for California in 2018 for 1-Year horizon and person survey
    if os.path.exists('data/ASCIncome.csv'):
        logging.info("ASCIncome dataset exists")
    else:
        ASC_fetcher(states=["CA"],
                    survey_year='2018',
                    horizon='1-Year',
                    survey='person',
                    file_name='data/ASCIncome.csv')
        logging.info("Downloading ASCIncome data for California in 2018 for 1-Year horizon and person survey")


def asc_data():
    features = ['SEX', 'RAC1P', 'AGEP', 'POBP', 'MAR', 'SCHL', 'OCCP', 'WKHP', 'COW']
    label = 'label'
    # Read ASCIncome 2018 dataset
    ASC = pd.read_csv("data/ASCIncome.csv")
    return ASC, features, label
