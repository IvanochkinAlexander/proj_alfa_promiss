import pandas as pd

def read_and_concat (list_of_files):

    all_df = pd.DataFrame()
    for file_name in list_of_files:
        print (file_name)
        temp = pd.read_json('~/portia_projects/output_data/{}.json'.format(file_name))
        print (temp.shape[0])
        temp.fillna(0, inplace=True)

        field_col = []
        for i in temp.columns:
            if i.startswith('field'):
                field_col.append(i)

        all_text = []
        for i in range(temp.shape[0]):
            temp_text = ''
            for k in field_col:
                str_value=str(temp.loc[i, k])
                if str_value != '0':
                    temp_text+=str_value
                    temp_text+='. '
            all_text.append(temp_text)

        final_df = pd.concat([temp[['url']], pd.DataFrame(all_text)],axis=1)
        final_df = final_df.rename(columns={0:'text'})
        all_df = pd.concat([all_df, final_df])
        all_df = all_df.reset_index(drop=True)
    return all_df

list_of_files = ['mb', 'open']

concated = read_and_concat(list_of_files)
concated.to_excel('../output/promiss/concated.xlsx', index=False)
