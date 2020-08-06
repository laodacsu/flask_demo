#!/usr/bin/rnv python
# -*- coding:utf-8 -*-
# authoor:zjd time:2020/3/14
import happybase
import numpy as np
import pickle
from flask import Flask
import json#引用json模块

from statsmodels.tsa.arima_model import ARMA
import warnings
warnings.filterwarnings("ignore")

connection = happybase.Connection(host="192.168.100.100")
app = Flask(__name__)

@app.route('/')
def index():
    table2 = connection.table('mytable')
    d = table2.row(b'1004')[b'cf1:id']
    model = pickle.loads(d, encoding='bytes')
    ans = model.forecast(1)[0]
    return json.dumps({"ans":ans})

def other():
    table_name_list = connection.tables()  # connection.tables()：获取Hbase实例中的表名列表，返回一个list
    print(table_name_list)
    table = connection.table('fruit')
    df = np.random.randn(200)


    # table.put(b'row-key', {b'family:qual1': b'value1',
    #                        b'family:qual2': b'value2'})
    #
    # row = table.row(b'row-key')
    # print(row[b'family:qual1'])  # prints 'value1'
    # table.put('1004',{b'info:color':'orange',
    #                    b'info:name':'orange'})
    #
    # for key, data in table.rows([b'1001', b'1002']):
    #     print(key, data)  # prints row key and data for each row
    #     for k,v in data.items():
    #         print(k,v)
    # connection.create_table(
    #     'mytable',
    #     {'cf1': dict(max_versions=10),
    #      'cf2': dict(max_versions=1, block_cache_enabled=False),
    #      'cf_ha': dict(),  # use defaults
    #     }
    # )
    # retrieve  检索
    table2 = connection.table('mytable')

    model = ARMA(df, order=(1,0)).fit(disp=False)
    dt = pickle.dumps(model)
    table2.put(b'1004',{'cf1:id':dt})

    # table2.put("1001",{b"cf1:id":"1",b"cf2:id":'1',b"cf3:id":'1'})
    for key, data in table2.scan():
        print(key,data)  # prints 'value1' and 'value2'
        for k,v in data.items():
            print(k,v)

    for ke,data in table.rows([b'1001',b'1002'],columns=['info']):
        print(ke,data)

    print("--------------------------")
    values = table.cells('1001', 'info:name', versions=2)
    for value in values:
        print((value))


    print('------------------------------')
    for key, data in table.scan(row_start=b'1001'):
        print(key, data)

    print('-----------------------------')
    for key, data in table.scan(row_stop=b'1003'):
        print(key, data)

    print('----------------------------')
    for key, data in table.scan(row_prefix=b'100'):
        print(key, data)
    # for key, data in table.scan(row_prefix=b'row'):
    #     print(key, data)  # prints 'value1' and 'value2'
    #
    # row = table.delete(b'row-key')
    print('-------------------------')
    d = table2.row(b'1004')[b'cf1:id']
    model = pickle.loads(d,encoding='bytes')
    ans = model.forecast(1)
    print(ans)

if __name__ == '__main__':
    app.run()