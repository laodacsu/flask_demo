#!/usr/bin/rnv python
# -*- coding:utf-8 -*-
# authoor:zjd time:2020/3/21

import pandas as pd
df = pd.DataFrame({'class':['A','A','A','B','B','B','C','C'],
                  'id':['a','b','c','a','b','c','a','b'],
              'value':[1,2,3,4,5,6,7,8]})
df.set_index(['class', 'id'],inplace=True)
print(df.to_json(orient='index'))