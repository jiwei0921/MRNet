# -*- coding: utf-8 -*-
import os
import pandas as pd
import natsort
import shutil
import sys
from tqdm import trange


def get_file_name_and_postfix(filenames, postfix, flag=False, filter_msg=None):
    NameList = []
    for file in filenames:
        if os.path.splitext(file)[1] in postfix:
            NameList.append(os.path.splitext(file)[0])
            _postfix = os.path.splitext(file)[1]
        else:
            print('Error: Illegal postfix!')
            sys.exit()

    if flag:
        NameList = natsort.natsorted(NameList)
        NameList = [x for x in NameList if _contains(x, filter_msg)]
    else:
        NameList = natsort.natsorted(NameList)

    return NameList, _postfix


def _contains(x, item):
    return x in item


def main(data_path, mask_path, Name, filter_flag=False, filter_msg=None):
    postfix = ['.png','.tif','.jpg','JPG','.bmp','.BMP']
    (data_dirpath, data_dirnames, data_filenames) = list(os.walk(data_path))[0]
    (mask_dirpath, mask_dirnames, mask_filenames) = list(os.walk(mask_path))[0]


    ImgName, Img_postfix = get_file_name_and_postfix(data_filenames, postfix, filter_flag, filter_msg)
    LabelName, Label_postfix = get_file_name_and_postfix(mask_filenames, postfix, filter_flag, filter_msg)


    try:
        if (ImgName == LabelName): print('Generate CSV files\n Initializing...')
    except Exception as e:
        print(f"No Match between img and Label\n"
              f"Run Error:{e}")

    DF_all = pd.DataFrame(columns=['imgName', 'maskName'])

    for imgNum in trange(len(ImgName)):
        DF_all.loc[imgNum, 'imgName'] = ImgName[imgNum] + Img_postfix
        DF_all.loc[imgNum, 'maskName'] = LabelName[imgNum] + Label_postfix

    DF_name = data_dirpath + '/'+ Name +'.csv'
    DF_all.to_csv(DF_name)
    shutil.move(DF_name, os.path.dirname(data_dirpath))
    print('Done...')



if __name__ == '__main__':
    cfg = dict(
        data_path='/Users/muscle/Desktop/Data/test_images',
        mask_path='/Users/muscle/Desktop/Data/test_masks',
        csv_name='SOD',
        filter_flag=False,
        filter_msg=None,
    )

    main(cfg['data_path'],cfg['mask_path'],cfg['csv_name'],cfg['filter_flag'],cfg['filter_msg'])


'''
FAQ:
Q1: You might get 'Error: Illegal postfix!', if your code run in the MAC OS.
A1: you need to check if data_path contains '.DS_Store' 
'''