---
layout: post
title: Fast Search Pandas DF
description: Project fast-search-pandas
image: assets/images/pic11.jpg
nav-menu: false
---

**This notebook is designed to improve your skills to search specific data inside a pandas DataFrame.**
Notebook available on kaggle too: https://www.kaggle.com/code/gipalm/fast-search-pandas
Very often in Data Science you need to search individual values inside a DataFrame that is not possible to do inside a join/merge function.

This notebook will help you.


1. Searching inside column in pandas
2. Searching inside index pandas
3. Using numpy.where
4. Using list(for index) and numpy(for data)
5. Using list(for index) and list(for data)
6. Using dictionary instead of lists and dataframes


```python
#Load DataFrame
import pandas as pd
df=pd.read_csv("/kaggle/input/open-problems-single-cell-perturbations/adata_obs_meta.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obs_id</th>
      <th>library_id</th>
      <th>plate_name</th>
      <th>well</th>
      <th>row</th>
      <th>col</th>
      <th>cell_id</th>
      <th>donor_id</th>
      <th>cell_type</th>
      <th>sm_lincs_id</th>
      <th>sm_name</th>
      <th>SMILES</th>
      <th>dose_uM</th>
      <th>timepoint_hr</th>
      <th>control</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000006a87ba75b72</td>
      <td>library_4</td>
      <td>plate_4</td>
      <td>F7</td>
      <td>F</td>
      <td>7</td>
      <td>PBMC</td>
      <td>donor_2</td>
      <td>T cells CD4+</td>
      <td>LSM-4944</td>
      <td>MLN 2238</td>
      <td>CC(C)C[C@H](NC(=O)CNC(=O)c1cc(Cl)ccc1Cl)B(O)O</td>
      <td>1.0</td>
      <td>24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000233976e3cd37</td>
      <td>library_0</td>
      <td>plate_3</td>
      <td>D4</td>
      <td>D</td>
      <td>4</td>
      <td>PBMC</td>
      <td>donor_1</td>
      <td>T cells CD4+</td>
      <td>LSM-46203</td>
      <td>BMS-265246</td>
      <td>CCCCOc1c(C(=O)c2c(F)cc(C)cc2F)cnc2[nH]ncc12</td>
      <td>1.0</td>
      <td>24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0001533c5e876362</td>
      <td>library_2</td>
      <td>plate_0</td>
      <td>B11</td>
      <td>B</td>
      <td>11</td>
      <td>PBMC</td>
      <td>donor_0</td>
      <td>T regulatory cells</td>
      <td>LSM-45663</td>
      <td>Resminostat</td>
      <td>CN(C)Cc1ccc(S(=O)(=O)n2ccc(/C=C/C(=O)NO)c2)cc1</td>
      <td>1.0</td>
      <td>24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00022f989630d14b</td>
      <td>library_35</td>
      <td>plate_2</td>
      <td>E6</td>
      <td>E</td>
      <td>6</td>
      <td>PBMC</td>
      <td>donor_0</td>
      <td>T cells CD4+</td>
      <td>LSM-43216</td>
      <td>FK 866</td>
      <td>O=C(/C=C/c1cccnc1)NCCCCC1CCN(C(=O)c2ccccc2)CC1</td>
      <td>1.0</td>
      <td>24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0002560bd38ce03e</td>
      <td>library_22</td>
      <td>plate_4</td>
      <td>B6</td>
      <td>B</td>
      <td>6</td>
      <td>PBMC</td>
      <td>donor_2</td>
      <td>T cells CD4+</td>
      <td>LSM-1099</td>
      <td>Nilotinib</td>
      <td>Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)...</td>
      <td>1.0</td>
      <td>24</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obs_id</th>
      <th>library_id</th>
      <th>plate_name</th>
      <th>well</th>
      <th>row</th>
      <th>col</th>
      <th>cell_id</th>
      <th>donor_id</th>
      <th>cell_type</th>
      <th>sm_lincs_id</th>
      <th>sm_name</th>
      <th>SMILES</th>
      <th>dose_uM</th>
      <th>timepoint_hr</th>
      <th>control</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>240085</th>
      <td>ffff28f274e983df</td>
      <td>library_27</td>
      <td>plate_0</td>
      <td>G12</td>
      <td>G</td>
      <td>12</td>
      <td>PBMC</td>
      <td>donor_0</td>
      <td>NK cells</td>
      <td>LSM-3349</td>
      <td>Mometasone Furoate</td>
      <td>C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C...</td>
      <td>1.0</td>
      <td>24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>240086</th>
      <td>ffff32893af5befb</td>
      <td>library_31</td>
      <td>plate_4</td>
      <td>E7</td>
      <td>E</td>
      <td>7</td>
      <td>PBMC</td>
      <td>donor_2</td>
      <td>T cells CD4+</td>
      <td>LSM-2287</td>
      <td>Midostaurin</td>
      <td>CO[C@@H]1[C@H](N(C)C(=O)c2ccccc2)C[C@H]2O[C@]1...</td>
      <td>1.0</td>
      <td>24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>240087</th>
      <td>ffff6c3e0a7b05ad</td>
      <td>library_38</td>
      <td>plate_1</td>
      <td>C5</td>
      <td>C</td>
      <td>5</td>
      <td>PBMC</td>
      <td>donor_2</td>
      <td>NK cells</td>
      <td>LSM-45786</td>
      <td>BAY 87-2243</td>
      <td>Cc1cc(-c2nc(-c3ccc(OC(F)(F)F)cc3)no2)nn1Cc1ccn...</td>
      <td>1.0</td>
      <td>24</td>
      <td>False</td>
    </tr>
    <tr>
      <th>240088</th>
      <td>ffff8e571c7e8cb0</td>
      <td>library_28</td>
      <td>plate_5</td>
      <td>B1</td>
      <td>B</td>
      <td>1</td>
      <td>PBMC</td>
      <td>donor_1</td>
      <td>B cells</td>
      <td>LSM-43181</td>
      <td>Belinostat</td>
      <td>O=C(/C=C/c1cccc(S(=O)(=O)Nc2ccccc2)c1)NO</td>
      <td>0.1</td>
      <td>24</td>
      <td>True</td>
    </tr>
    <tr>
      <th>240089</th>
      <td>ffffe67500d95d8d</td>
      <td>library_9</td>
      <td>plate_3</td>
      <td>E1</td>
      <td>E</td>
      <td>1</td>
      <td>PBMC</td>
      <td>donor_1</td>
      <td>Myeloid cells</td>
      <td>LSM-43181</td>
      <td>Belinostat</td>
      <td>O=C(/C=C/c1cccc(S(=O)(=O)Nc2ccccc2)c1)NO</td>
      <td>0.1</td>
      <td>24</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



We are going to search inside variable obs_id for the exactly same value with some different methods and see which one is the best.


```python
%%timeit -n 100
#First method - Filtering your dataset

df[df['obs_id']=='0002560bd38ce03e']
```

    12.5 ms ± 206 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    

Not bad, but I think we can improve this.


```python
#Set variable you need to search on index
df.set_index('obs_id',inplace=True)
```

**set_index(column)** -> changes the index of the dataframe (use .head() to see what it looks like)
**inplace=True** means that you are reffering to the same dataframe (df) to output the value.

*it's the same as df=df.set_index('obs_id')*


```python
%%timeit -n 100
#Second method - Searching using indexed dataframe
df.loc['0002560bd38ce03e']
```

    The slowest run took 8.40 times longer than the fastest. This could mean that an intermediate result is being cached.
    99.2 µs ± 123 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    

What a great improvement, let's 


```python
import numpy as np
index=np.array(df.index)
values=np.array(df.values)
```


```python
%%timeit
#Third method - Using numpy where
np.where(index=='0002560bd38ce03e')
```

    3.68 ms ± 17.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    


```python
list_index=list(index)
list_values=values.tolist()
```


```python
values[list_index.index('0002560bd38ce03e')]
```




    array(['library_22', 'plate_4', 'B6', 'B', 6, 'PBMC', 'donor_2',
           'T cells CD4+', 'LSM-1099', 'Nilotinib',
           'Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1',
           1.0, 24, False], dtype=object)




```python
%%timeit
#Fourth method - Index is inside list on python and the data is on numpy data
values[list_index.index('0002560bd38ce03e')]
```

    184 ns ± 2.2 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    


```python
%%timeit
#Fifth method - Index is inside list on python and the data is on another list
list_values[list_index.index('0002560bd38ce03e')]
```

    110 ns ± 0.491 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    

Is it the same value if we look for a random element in the list?


```python
df.sample(1).index
```




    Index(['70611ad019b262a3'], dtype='object', name='obs_id')




```python
%%timeit
#Fifth method - Index is inside list on python and the data is on another list
list_values[list_index.index('34dc16ad9c7050d7')]
```

    603 µs ± 4.27 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    

Let's review the second method with the same index


```python
%%timeit -n 100
#Second method - Searching using indexed dataframe
df.loc['34dc16ad9c7050d7']
```

    56.2 µs ± 8.41 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    

For the random sample the second method is beating the fifth method


```python
dict_df=df.to_dict(orient='index')
```


```python
%%timeit
#Sixth method - Using dictionary instead of lists and dataframes
dict_df['34dc16ad9c7050d7']
```

    30.9 ns ± 0.808 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    


```python
%%timeit
#Comparing with get function of dictionaries does not improve
dict_df.get('34dc16ad9c7050d7')
```

    41.8 ns ± 2.4 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    

**Conclusion**

If you convert your dataframe for a dictionary and use the index as key, you can get 1000x faster lookup/search on your dataset.

This is only useful if you have to search a lot of times to get the max optimization possible.

Below is all you need to use.


```python
import pandas as pd

#Load your dataset 
df=pd.read_csv("/kaggle/input/open-problems-single-cell-perturbations/adata_obs_meta.csv")

#Use the column (obs_id) that you want to search
df.set_index('obs_id',inplace=True)

#Create a dictionary of the dataframe
dict_df=df.to_dict(orient='index')

#Search
dict_df['34dc16ad9c7050d7']
```




    {'library_id': 'library_46',
     'plate_name': 'plate_1',
     'well': 'E6',
     'row': 'E',
     'col': 6,
     'cell_id': 'PBMC',
     'donor_id': 'donor_2',
     'cell_type': 'T cells CD4+',
     'sm_lincs_id': 'LSM-43216',
     'sm_name': 'FK 866',
     'SMILES': 'O=C(/C=C/c1cccnc1)NCCCCC1CCN(C(=O)c2ccccc2)CC1',
     'dose_uM': 1.0,
     'timepoint_hr': 24,
     'control': False}


