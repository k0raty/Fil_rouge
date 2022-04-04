import xlrd
import numpy as np
import os

workbook = xlrd.open_workbook(os.getcwd()+'\\Dataset\\2_detail_table_customers.xls')
SheetNameList = workbook.sheet_names()
for i in np.arange( len(SheetNameList) ):
    print(SheetNameList)