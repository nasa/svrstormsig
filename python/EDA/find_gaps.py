'''
Created on Nov 20, 2019

@author: ahossein
'''
'''
Created on Nov 1, 2019

@author: ahossein
'''
import numpy as np
import pandas as pd
import os
import numpy as np
import ast
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def listdir_nohidden(path):
  '''
  Obtains list of files/folders within a directory, while excluding any hidden files. 
  Args:
    path: Path to directory to create list from.
  Output:
    A list of files and folders within a given path if they are not hidden files or folders.
  '''
  listings = []
  for f in os.listdir(path):
      if not f.startswith('.'):
          listings.append(f)
  return listings



'''Import data frame'''

def initial_df(csv_path):
  '''
  Imports the update_rdr csv file as a pandas dataframe. 
  removes rows that do not have associated VIS or IR images.
  VIS column entries conveted to the png file format.
  Converts the alphanumeric time column from string to int.
  Args:
    csv_path: Read Data from a netcdf file or IR Data
  Output:
    Pandas data frame based on update_rdr csv for a given day with the no missing VIS or IR files,
    a time column with int values, and a VIS column with image names that reflect the combinednoplume
    image names which have a .png extention. 
  '''
  df = pd.read_csv(csv_path)
  dfIRDrop = df[df.IR_Data_File != "NO_VALID_FILE"]
  df = dfIRDrop[dfIRDrop.VIS_Data_File != "NO_VALID_FILE"]
  #convert viz file name from nc to png.
  df.VIS_Data_File = df.VIS_Data_File.str.strip().str.lower().str.replace('vis_', '').str.replace('.nc', '.png')
  #convert time from str to int.
  import re
  df['Time'] = df['Time'].str.replace('Z|T', "")
  df['Time'] = df['Time'].astype(int)
  return df


def find_gaps(dfz):
  '''
  Gets unique time
  selects only daytime images
  finds the time passed between each time column entry
  Removes trailing zeros from each timestamp.
  Args:
    dfz: dataframe from output of initial_df function (see above)
  Output:
    pandas data frame with unique time values associated with a SZA test of 1
    and a TimeDiff column with time elapsed for each unique time value.  
    
  '''
  
  dfz.drop_duplicates(subset ="Time", 
                       keep = "first", inplace = True)
  dfz = dfz.sort_values(by=['Time'])
  dfDayTime = dfz[(dfz['sza_test'] == 1)]

  #optional description of time
  #dfDayTime["Time"].describe().apply(lambda x: format(x, 'f'))

  dfDayTime= dfDayTime.sort_values(by=['Time'])

  dfDayTime['TimeDiff'] = dfDayTime.Time.diff()

  #Creating a time column devided by 100
  dfDayTime["TimeDiv100"]= dfDayTime.Time.div(100).apply(lambda x: format(x, 'f'))
  
  return dfDayTime

def gap_exceptions(dfz):
  '''
  Removes common common gaps that are due to passage of time, but are due to changes in minutes, hours, 
  and days from the list of gaps. There is slight risk that gaps may go undetected on the rare chance that a
  gap in the sattelite feed is exactly 41 minutes. Need to convert to pandas date-time format to avoid this.
  Args:
    dfz: dataframe resulting from the find_gaps function
  Output:
    Pandas data frame with consisting of only significant gaps in satellite feed.     
  '''  
  
  gaps_ = dfz[(dfz['TimeDiff'] != 100)]           
  '''one minute'''
  gaps_ = gaps_[(gaps_['TimeDiff'] != 4100)]      
  ''' next hour, ex: 1:59 -> 2:00 '''
  gaps_ = gaps_[(gaps_['TimeDiff'] != 764100)]    
  ''' Next day'''

  # Change this from Time to VIS_file_name to get that range
  return gaps_

def file_name(column_name, dfDayTime, gaps_): # fix func name!
  '''
  Generates a list of values (from a given column (column_name))   
  associated with gaps in the feed. For the GCP AutoML CSV's to be
  generated properly the column_name should be the Time column..
   
  Args:
    column_name: data frame resulting from the find_gaps function
    dfDayTime: data frame from the find_gaps function
    gaps_: df from the gap exceptions function.
  Output:
    A simple list of values with a given column associated with the gaps in the feed.
  '''
  file_name_column_number = gaps_.columns.get_loc(column_name)
  #print(dfDayTime.VIS_Data_File)
  lenindex = len(gaps_.index)
  #print(lenindex)
  maxlen = len(dfDayTime.index)
  #print('Selections starts at'+ gaps_.iloc[i,file_name_column_number])
  i = 0
  GAPDATES = []
  if lenindex > 1:
      for i in range (0,(lenindex-1)):
          print(str(gaps_.iloc[i,file_name_column_number]) , "  ---->  " , gaps_.iloc[i+1,file_name_column_number])
          GAPDATES.append(gaps_.iloc[i+1,file_name_column_number])
          if i == (lenindex-2):
              print(gaps_.iloc[i+1,file_name_column_number] , "  ---->  " , dfDayTime.iloc[maxlen-1, file_name_column_number])
          else:
              print('Error: Check Code')
  else:
      print('Usable image range begins with file: ', gaps_.iloc[i,file_name_column_number])
      print('...And no gaps were detected.')  
  print(GAPDATES)  
  return GAPDATES

def main():
  
  date_dict = {'2017087': ['20170328',[['2017087_203056.png','2017087_235056.png',(308,308),'A']]],
               '2017095': ['20170405',[['2017095_122026.png','2017095_180226.png',(300,500),'A'],
                                      ['2017095_183031.png','2017095_224631.png',(650,450),'B']]],
               '2017136': ['20170516',[['2017136_183456.png','2017136_204556.png',(220,70),'A'],
                                       ['2017136_204556.png','2017137_001756.png',(850,75),'B']]],
               '2017138': ['20170518',[['2017138_180526.png','2017138_202926.png',(400,760),'A'],
                                       ['2017138_204226.png','2017139_001326.png',(400,760),'B']]],
               '2017179': ['20170628',[['2017179_180027.png','2017180_002527.png',(400,760)]]],
               '2017180': ['20170629',[['2017180_180057.png','2017181_002257.png',(500,100),'A']]]
            }
  for jul_date, ctl_lst in date_dict.items():  
    csv_path = '../../../data/'+ctl_lst[0]+'/update_rdr_data.csv'
  
    df= initial_df(csv_path)
    dfz = df.copy()
    dfgaps = find_gaps(dfz)
    #print(dfgaps)
  
    dfgaps2 = gap_exceptions(dfgaps)
    #print(dfgaps2)
    print('\n')
    print('DAY: ',ctl_lst[0])
    column_name = 'VIS_Data_File'
    gap_list = file_name(column_name, dfgaps, dfgaps2)
    

    #print (gap_list)
    
        
if __name__ == '__main__':
    main()