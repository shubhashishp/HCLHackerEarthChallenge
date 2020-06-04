# -*- coding: utf-8 -*-
"""

@author: shubhashish.p
"""

import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import json

var_nan = "nan"

import glob
files = glob.glob("*.txt")
filecount = (len(files))

#create complete data
data = pd.DataFrame()
for file in files:
  #print(file)
  file_lines = open(file, "r")    
  lines = file_lines.readlines() 
  count = 0
# Strips the newline character 
  for line in lines: 
   
#    print(line.strip()) 
#    print("Line{}: {}".format(count, line.strip()))
    x = line.replace("\n","")
    x = x.replace(" ","")
    if x.strip() != "":
        count +=1    
        a_row = pd.Series([file,count,line])
        row_df = pd.DataFrame([a_row])
        data = pd.concat([data,row_df], ignore_index=False)
    
data = data.rename(columns={0: 'file',1:'No',2:'line'})

data['group_word']=""
data['group_word'] = data['line'].apply(lambda element: str(re.split(r'\s\s+',element)))

data['group_word'] = data['group_word'].apply(lambda element: element.replace("['',","["))
data['group_word'] = data['group_word'].apply(lambda element: element.replace(", '']","]"))
data['group_word'] = data['group_word'].apply(lambda element: element.replace(", '' ,",","))
data['token_count'] = data['line'].apply(lambda element: len([i for i in re.split(r'\s\s+',element) if i]))

data['year_freq'] = 0
data['isHeader_part'] = 0

data['year_freq'] = data['group_word'].apply(lambda element: len(re.findall(r"('20[1-2]\d')+",element)) )
#data['year_freq'] = data['group_word'].apply(lambda element: len(re.findall(r"\s*Notes\s*",element)) )
data['isHeader_part'] = data['group_word'].apply(lambda element: len(re.findall(r"\s*Â£\s*",element)) )

data['isContent'] = 0
data['isContent'] = data['group_word'].apply(lambda element: len(re.findall(r'\'\(?\d*?,*\d+.?\d*\)?\'\s?]',element)) )
data['phrase'] = data['line'].apply(lambda element: getPhrase(element))
phrase_as_content_list = data['phrase'][data.isContent > 0].value_counts()[:60].rename_axis('unique_values').reset_index(name='counts')
word_list = phrase_as_content_list['unique_values'].tolist()
word_list = [ elem for elem in word_list if len(elem) > 5]

data['isContent'][data.isContent == 0] = data['phrase'][data.isContent == 0].apply(lambda element: 1 if element.upper() in (word.upper() for word in word_list) else 0)

data['value'] = 0

def getPhrase(text):
    list = [i for i in re.split(r'\s\s+',text) if i]
    print(list)
    if len(list) > 0 :
        return str(list[0]).replace('Â£','&#163').replace('Â€','&#8364').replace('Â$','&#36')
    else:
        return var_nan


def extractValue(val):
    val = val.replace(",","");
    isbrack = len(re.findall(r'\(\d+\)?',val))
    if isbrack==1:
        val = val.replace("(","-");
        val = val.replace(")","");
    isValDot = len(re.findall(r'.',val))
    if isValDot > 0:
        parts=val.split(".")
        val = parts[0] 
    return val
        

def getValue(file, line_text):
    print(file)    
    #get header
    header = getHeader(file)
    print(header)
    if header["isHeader"] == False :
        return var_nan
    
    #header exist    
    #{'isHeader','isNotes','is_19','left','char_index','tokens', 'max_count': 2}
    
    if header['is_19']:
        print("2019 exists")
        line_list =  [i for i in re.split(r'\s\s+',line_text) if i] 
        line_tokens = len(line_list)
        print(line_list)
        #####
        
        if line_tokens == 1:
            return var_nan
        else:
            value_after_19 = header['tokens'] - header['left'] - 1
            if(line_tokens >= value_after_19 ):
                 val = extractValue(line_list[line_tokens-value_after_19-1])
                 return val
            else:
                 return var_nan         
            
        #####       
    else:
        return var_nan
    
    #header not exist
    
    #get value
    
def getHeader(file):
    #get header line    
    bs_header={}
    data_header = data[(data.file==file) & (data.year_freq > 0)].nsmallest(1,'No')
    if len(data_header) > 0:
        bs_header["isHeader"] = True
    else:
        find_excep_headers(file)
        data_header = data[(data.file==file) & (data.year_freq > 0)].nsmallest(1,'No')
        if len(data_header) > 0:
            bs_header["isHeader"] = True
        else:
            bs_header["isHeader"] = False
            return bs_header
        
    w_list = [i for i in re.split(r'\s\s+',data_header.line[0]) if i]
    #if header have notes
    #notes_len = len(re.findall(r"('Notes')",data_header.group_word[0]))
    if 'Notes' in w_list:
        bs_header["isNotes"] = True
    else:
        bs_header["isNotes"] = False
        
    #if header have 2019
    
    if '2019' in w_list:
        bs_header["is_19"] = True
        bs_header['left'] = w_list.index('2019')
        bs_header['char_index'] = data_header.line[0].index('2019')
    elif '(2019)' in w_list:
        bs_header["is_19"] = True
        bs_header['left'] = w_list.index('(2019)')
        bs_header['char_index'] = data_header.line[0].index('(2019)')
    else:
        bs_header["is_19"] = False
        bs_header['left'] = -1
        bs_header['char_index'] = -1
    
    #tokens
    bs_header['tokens'] = len(w_list)    
    
    #max Tokens for content
    bs_header['max_count'] = data[(data.file==file) & (data.isContent > 0) & (data.year_freq == 0)]['token_count'].max()
             
    return bs_header
    #if number have 2 find if have notes
    #or appeared first and have nothing apart from year or followed content lines

def getOutcome():
    fill_value()
    output_data = pd.DataFrame()
    output_data["Filename"] = files
    data['bsRow'] = 0
    
    output_data["Extracted Values"] = output_data['Filename'].apply(lambda element: str(json.dumps(getDict(element))).replace('"\n": "nan"',''))
    output_data["Filename"] = output_data['Filename'].apply(lambda element: element.replace(".txt",""))
    #output_data.to_csv("C:\\Shubhashish\\Experiments\\Hackethon\\MLChallenge\\f0230b3e675e11ea\\HCL ML Challenge\\FinalResult.csv", sep=',',index=False)
    output_data.to_csv("FinalResult.csv", sep=',',index=False)
    

def getDict(file):
    min_ind = data[(data.file == file) & (data.isContent==1) & (data.year_freq == 0)]['No'].min()
    max_ind = data[(data.file == file) & (data.isContent==1) & (data.year_freq == 0)]['No'].max()
    data['bsRow'][(data.file == file) & (data.No >= min_ind) & (data.No <= max_ind)] = 1
    
    year_min_ind = data[(data.file == file) & (data.isContent==1) & (data.year_freq > 0)]['No'].min()
    data['bsRow'][(data.file == file) & (data.No > year_min_ind) & (data.No <= max_ind) & (data.isHeader_part == 0)] = 1
    dict_val =  data[(data.file == file) & (data.bsRow==1) & (data.year_freq == 0)].set_index('phrase')['value'].to_dict()
    print(dict_val)
    #dict_val.replace()
    return dict_val

def fill_value():
    data['value'] = var_nan
    data['value'][(data.isContent==1) & (data.year_freq ==0 )] = data[(data.isContent==1) & (data.year_freq ==0 )].apply(lambda element: getValue(element['file'],element['line']), axis=1)
    
    
def find_excep_headers(file):
    data_header = data[(data.file==file) & (data.year_freq > 0)].nsmallest(1,'No')
    year_min_ind = data[(data.file == file) & (data.isContent==1)]['No'].min()
    #can use isHeader_part
    ind = year_min_ind
    if year_min_ind > 2:
        ind = year_min_ind -2
        h_line = data[(data.file == file) & (data.No == ind )]['line']   
        
        new_line = re.sub("\d{2}\s[a-zA-Z]{3}\s2019", "2019",h_line[0])
        data['line'][(data.file == file) & (data.No == ind)] = new_line
    
        data['year_freq'][(data.file == file) & (data.No == ind)] = len(re.findall(r"20[1-2]\d",h_line[0]))

getOutcome()


