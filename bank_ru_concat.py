import pandas as pd
import os


def concat_in_directory ():
    
    """concating data in directory"""
    
    for i in sorted(os.listdir('../output/promiss/')):

        if i.startswith('submisssion_bank'):
            print(i)
            temp = pd.read_excel('../output/promiss/{}'.format(i))
            all_df = pd.concat([all_df, temp])
    first_df = pd.read_excel('/root/portia_projects/output_data/bank_ru_processed.xlsx')
    all_df.iloc[:,25:].to_excel('../output/promiss/temp_classified.xlsx', index=False)
    print (first_df.shape[0])
    print (all_df.shape[0])
    first_df = pd.concat([first_df, all_df[['max_value0']]],axis=1)
    first_df.to_excel('../output/promiss/bank_ru_classified.xlsx', index=False)
    files_list = ['submisssion_bank10_33_1_660738.xlsx','submisssion_bank10_49_42_753923.xlsx','submisssion_bank11_6_49_60601.xlsx','submisssion_bank11_23_23_635542.xlsx','submisssion_bank11_40_2_174756.xlsx']

all_df = pd.DataFrame()

for i in files_list:
    print (i)
    temp = pd.read_excel('../output/promiss/{}'.format(i))
    all_df = pd.concat([all_df, temp])

all_df.iloc[:,25:].to_excel('../output/promiss/temp_classified.xlsx', index=False)
