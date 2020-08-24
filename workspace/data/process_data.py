import sys
import pandas as pd
from sqlalchemy import create_engine

def custom_split(x):
    """
    Expected Input: String of format 'sometext-somenumber'.
    Output: Returns 'somenumber' in string format.
    """
    split_str = x.split('-')
    number = split_str[1]
    return number

def convert_num(x):
    """
    Parse the input string into an integer.
    """
    return int(x)


def load_data(messages_filepath, categories_filepath):
    """
    This function reads messages data from the messages_filepath.
    It also reads corresponding categories data from the categories_filepath.
    It performs various ETL steps to ensure these two data can be merged.
    It returns the transformed df.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id', how='outer')
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    row = categories.iloc[0].str[:-2]
    category_colnames = row
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(custom_split)
        # convert column from string to numeric
        categories[column] = categories[column].apply(convert_num)
        
    df = df.drop('categories', axis=1)
    df = pd.concat([df,categories], axis=1)
    return df


def clean_data(df):
    """
    Cleans up the input df by removing duplicate rows.
    """
    # check number of duplicates
    df_duplicates = df[df.duplicated()]
    print("Number of duplicates: ", df_duplicates.shape[0])
    # drop duplicates
    print("Removing all duplicates")
    df = df.drop_duplicates()
    # check number of duplicates
    df_duplicates = df[df.duplicated()]
    print("Number of duplicates after removal: ", df_duplicates.shape[0])
    return df


def save_data(df, database_filename):
    """
    After all ETL operations are performed, thie function
    saves the clean df into a local SQLite db file which can then be
    loaded and read from.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False)
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
