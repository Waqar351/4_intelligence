import pandas as pd

def sector(data):

    #.....................Commercial.....................................................................................................................
    com = pd.melt(data, id_vars=['data_tidy'], value_vars=['com_co', 'com_n','com_ne','com_s','com_se'],
                            var_name='Regions', value_name='Energy_consumption(Gwh)')

    com = com.dropna()
    com["Regions"].replace({"com_co": "Midwest", "com_n": "North","com_ne": "Northeast","com_s": "South","com_se": "Southeast" }, inplace=True)
    
    com['Sectors'] = "Commercial"   # Making column with sector name in all rows
  
    #........................Industrial............................................................................................................................
    ind = pd.melt(data, id_vars=['data_tidy'], value_vars=['ind_co', 'ind_n','ind_ne','ind_s','ind_se'],
                            var_name='Regions', value_name='Energy_consumption(Gwh)')

    ind = ind.dropna()
    ind["Regions"].replace({"ind_co": "Midwest", "ind_n": "North","ind_ne": "Northeast","ind_s": "South","ind_se": "Southeast" }, inplace=True)
    ind['Sectors'] = "Industrial"

    #..................Residential.......................................................................................................................
    res = pd.melt(data, id_vars=['data_tidy'], value_vars=['res_co', 'res_n','res_ne','res_s','res_se'],
                            var_name='Regions', value_name='Energy_consumption(Gwh)')

    res = res.dropna()
    res["Regions"].replace({"res_co": "Midwest", "res_n": "North","res_ne": "Northeast","res_s": "South","res_se": "Southeast" }, inplace=True)
    
    res['Sectors'] = "Residential"

    df_all_sector = pd.concat([com,res, ind], ignore_index=True)
   
    return df_all_sector
