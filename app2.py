from tracemalloc import start
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import streamlit as st 
import datetime
from dateutil.relativedelta import relativedelta
import json 
from py2neo import Graph
import time 
import cv2
import datetime
import pickle
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from pathlib import Path

import streamlit_authenticator as stauth
#days_dict= {"Monday":0, "Tuesday":0, "Wednesday":0, "Thursday":0, "Friday":0, "Saturday":0, "Sunday":0}

def most_viewed(chapter_name, progress_df, topn=5): #Most viewed vids of a chapter (add class as well)
  chapter_df = progress_df[progress_df["container_name"] == chapter_name]
  x = chapter_df["content_title"].value_counts().nlargest(topn)
  return x

def get_completion_factor(no_of_topics, student_progress, test_percentage):
  x = no_of_topics
  y = student_progress
  z = test_percentage / 100
  factor = (z + y**2) / 2
  return factor
    
def skill_gap(content_repository, name, chapter, progress_df, contents):
  videos = return_topics(chapter, content_repository)
  no_of_topics = len(videos.keys())
  x = progress_df["percent_completion"][progress_df["container_name"] == chapter][progress_df["user_name"] == name].values
  score = progress_df["quiz_marks"][progress_df["container_name"] == chapter][progress_df["user_name"] == name].values
  scores = np.sum(score)/len(score)
  student_progress = np.sum(x)/len(x)
  student_progress /= 100
  lv_df, chapter_lv, student_lv = get_lv(name, chapter, progress_df)
  completion_factor = get_completion_factor(no_of_topics, student_progress, scores)
  sg = student_lv * completion_factor
  sg = 1/sg 
  skill_gap = (1 + np.exp(-1 * sg))
  return skill_gap

def return_topics(chapter_name, content_repository): 
  dict_topics = {} 
  for val in content_repository.values(): 
    (id, taxo, name, url, time) = val 
    subtopic = name 
    strs = taxo 
    idx = strs.rfind('>>')
    topic = strs[idx+2:]
    idx2 = strs.rfind('>>', 0, idx - 1)
    chapter = strs[idx2 + 2 : idx]
    if(chapter_name == chapter): 
      if(topic in dict_topics.keys()): 
        dict_topics[topic].append([id, topic, subtopic, url, time])
      else:
        dict_topics[topic] = []
        dict_topics[topic].append([id, topic, subtopic, url, time])
  for k, lis in dict_topics.items(): 
      l = sorted(lis,key=lambda x: (x[0]))
      dict_topics[k] = l 

  return dict_topics

def get_topics_json(class_, chapter_name, contents): 
  dicts = findkeys(contents[class_], chapter_name)
  subtopic = {}
  chapter_dict = dict(list(dicts)[0])
  for k in list(chapter_dict.keys()): 
    if(k == 'type'):
      continue
    subtopic_list = []
    if isinstance(chapter_dict[k], dict) and len(chapter_dict[k]) > 0:
      for st in list(chapter_dict[k].keys()): 
        if(st == 'type'): 
          continue 
        subtopic_list.append(st)
    subtopic[k] = subtopic_list
  topics = list(subtopic.keys())
  return topics, subtopic 

def findkeys(node, kv):
  if isinstance(node, list):
    for i in node:
      for x in findkeys(i, kv):
        yield x
  elif isinstance(node, dict):
    if kv in node:
      yield node[kv]
    for j in node.values():
      for x in findkeys(j, kv):
        yield x

def get_lv(student_name, chapter_name, progress_df):# Returns lv of all contents of a chapter and lv of chapter itself
  student_df = progress_df[progress_df["user_name"] == student_name]
  lv_df = student_df[student_df["container_name"] == chapter_name][["content_title", "learning_vel_content"]]
  student_lv = np.mean(student_df["learning_vel_content"].values)
# print("#lv#")
  # print(progress_df["learning_vel_content"][progress_df["user_name"] == student_name][progress_df["container_name"] == chapter_name].values)
  # print(progress_df["quiz_marks"][progress_df["container_name"] == chapter_name][progress_df["user_name"] == student_name].values)
  # print("###")
  clv1 = np.mean(progress_df["learning_vel_content"][progress_df["user_name"] == student_name][progress_df["container_name"] == chapter_name].values)
  clv2 = np.mean(progress_df["quiz_marks"][progress_df["container_name"] == chapter_name][progress_df["user_name"] == student_name].values)
  chapter_lv = (clv1+clv2)/2
  return lv_df, chapter_lv, student_lv

#function to convert time from secs to hours and mins 
def convert(seconds):
  if(time.gmtime(seconds).tm_hour > 0): 
    return time.strftime("%H hours, %M minutes", time.gmtime(seconds))
  else:
    return time.strftime("%M minutes", time.gmtime(seconds))

def findtime(url):
  # create video capture object
  data = cv2.VideoCapture(url)
  
  # count the number of frames
  frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
  fps = data.get(cv2.CAP_PROP_FPS)
  
  # calculate duration of the video
  seconds = round(frames / fps)
  return seconds

def get_timetable(content_repository, student_name, chapter_name, progress_df, content_df, contents, day_names,days_dict,num_hrs=0, days=7):# Creates a timetable for the studentfor next week
  student_df = progress_df[progress_df["user_name"] == student_name]
  topics_df = student_df[student_df["container_name"] == chapter_name][["class_id", "content_title", "percent_completion", "expected_time", "retention rate"]]
  
  content = get_content_t(chapter_name, progress_df, content_df)
  list_topics_main = []
  list_topics = []
  list_urls = []
  list_times = []

  for k,v in return_topics(chapter_name, content_repository).items(): 
    for video in v:
      list_topics_main.append(video[1])
      list_topics.append(video[2])
      list_urls.append(video[3])
      list_times.append(video[4])
  
  exp_times = np.zeros(len(list_topics))
  for idx in range(len(list_topics)): 
    t = list_topics[idx]
    if(t in content.keys()): 
      exp_times[idx] = content[t]
  names = list_topics
  final_timetable = {
                    "Monday": [], 
                    "Tuesday": [], 
                    "Wednesday": [], 
                    "Thursday": [], 
                    "Friday": [], 
                    "Saturday": [], 
                    "Sunday": []
                    }
  final_timetable2 = {
                    "Monday": [], 
                    "Tuesday": [], 
                    "Wednesday": [], 
                    "Thursday": [], 
                    "Friday": [], 
                    "Saturday": [], 
                    "Sunday": [],
                    "Give more time to complete the following topics": []
                    }
  keys = list(final_timetable.keys()) 
  keys2 = list(final_timetable2.keys()) 
  #Make 2 2D arrays one with name one with time, allot watch day 3 hrs move forward accordingly
  sg = skill_gap(content_repository, student_name, chapter_name, progress_df, contents)
  # uncomment and multiply with each time 
  # print('skill gap', sg)
  number = math.ceil(np.sum(len(exp_times)) / days)
  t2 = {}
  c = 0
  overall_time = 0
  # print(exp_times)
  for id in range(len(names)):
    n = names[id] 
    if n in topics_df["content_title"].values and topics_df["percent_completion"][topics_df["content_title"] == n].values[0] >= 0:
      percent_completed = topics_df["percent_completion"][topics_df["content_title"] == n].values[0]/100
      exp_times[id] = exp_times[id] * 1.25 * (1 - min(0.75, percent_completed)) 
      recall_rate = (100 - topics_df["retention rate"][topics_df["content_title"] == n].values[0]) / 100 
      exp_times[id] = exp_times[id] * recall_rate 
      exp_times[id] = exp_times[id] * sg
    else: 
      exp_times[id] = list_times[id] * 1.25 
      exp_times[id] = exp_times[id] * sg
    
  print("days gtt - ")
  print(days)
  for t in range(len(exp_times)): 
    time = math.ceil(exp_times[t])
    mins = time / 60
    rounded_off = math.ceil(mins / 10) * 10
    exp_times[t] = rounded_off * 60 
  overall_time = sum(exp_times)
  print("overall ",overall_time)
  print("num hrs", num_hrs)
  if(overall_time/3600>num_hrs):
    #student should devote more hours
    idx = 0 
    d = 0
    time_today = 0 
    total_time = sum(exp_times)
    #time_per_day = total_time/days
    while(d < days and idx < len(names)): 
      #if(days_dict.get(days[d]) - time_today >= 10):
      print("day name -",days_dict.get(day_names[d]))
      if(days_dict.get(day_names[d])*3600 > time_today): 
        final_timetable2[keys2[d]].append([list_topics_main[idx], names[idx], convert(exp_times[idx]), list_urls[idx]])
        time_today = time_today + exp_times[idx]
        idx = idx + 1 
      else: 
        time_today = 0 
        d = d + 1
    print("idx ", idx, "names", len(names))
    
    while(idx<len(names)):
      final_timetable2["Give more time to complete the following topics"].append([list_topics_main[idx], names[idx], convert(exp_times[idx]), list_urls[idx]])
      idx+=1
    
    day_names = np.append(day_names,"Give more time to complete the following topics")
    print(final_timetable2)
    t2 = {}
    c = 0
    print("day names",day_names)
    print("keys",list(final_timetable2.keys()))
    for i in list(final_timetable2.keys()):
      if (len(final_timetable2[i]) == 0):
        final_timetable2.pop(i)
        print("if i",i)
      else:
        print("else i",i)
        t2[day_names[c]] = final_timetable2[i]
        c+=1
    print(t2)
    done_before = topics_df[['content_title', 'expected_time', 'percent_completion']].values
    print(f"Topics done before: {done_before}")
    final_timetable_df = pd.DataFrame.from_dict(t2, orient='index')
    return final_timetable_df.transpose(), overall_time, t2
  else:
    #can easily complete the course
    idx = 0 
    d = 0
    time_today = 0 
    total_time = sum(exp_times)
    #time_per_day = total_time/days
    while(d < days and idx < len(names)): 
      #if(days_dict.get(days[d]) - time_today >= 10):
      print("day name -",days_dict.get(day_names[d]))
      if(days_dict.get(day_names[d])*3600 > time_today): 
        final_timetable[keys[d]].append([list_topics_main[idx], names[idx], convert(exp_times[idx]), list_urls[idx]])
        time_today = time_today + exp_times[idx]
        idx = idx + 1 
      else: 
        time_today = 0 
        d = d + 1 
    print(final_timetable)
    t2 = {}
    c = 0
    for i in list(final_timetable.keys()):
      if (len(final_timetable[i]) == 0):
        final_timetable.pop(i)
      else:
        t2[day_names[c]] = final_timetable[i]
        c+=1
    print(t2)
    done_before = topics_df[['content_title', 'expected_time', 'percent_completion']].values
    print(f"Topics done before: {done_before}")
    final_timetable_df = pd.DataFrame.from_dict(t2, orient='index')
    return final_timetable_df.transpose(), overall_time, t2 


def get_content_t(chapter_name, progress_df, content_df):
  p = set(content_df["content_title"][content_df["syllabus_name"] == chapter_name][content_df["group_category_name"] == "Learn"].values)
  content = {}
  for i in p:
    content[i] = np.nan
  x = progress_df[["content_title", "expected_time"]][progress_df["container_name"] == chapter_name].values
  s = set(progress_df["content_title"][progress_df["container_name"] == chapter_name].values)
  for i in x:
    content[i[0]] = i[1]
  avg = 0
  for i in list(content.values()):
    avg += np.nan_to_num(i)
  if(len(s) > 0): 
    avg = avg/len(s)
  for i in list(content.keys()):
    if(np.isnan(content[i]) or content[i] == 0):
      content[i] = avg
  
  return content


def convert_str_to_date(date_time):
  format = '%Y-%m-%d %H:%M:%S'
  datetime_str = datetime.datetime.strptime(date_time, format)
  return datetime_str

def calc_decay_rate(t): 
  c = 5
  k = 1.84
  b = (100 * k) / (np.log(t) ** c + k)
  return b 


def change_start_time(username, months, progress_df): 
  n = months
  updated_dates = []
  zdf = progress_df.loc[progress_df['user_name'] == username]
  for i, d in enumerate(zdf['start_time']): 
    if(d != d):
      continue
    else:
      dat = convert_str_to_date(d)
      new_dat = dat - relativedelta(months=n)
      date_format = '%Y-%m-%d %H:%M:%S'
      new_date_str = new_dat.strftime(date_format)
      updated_dates.append(new_date_str)
  progress_df.loc[progress_df['user_name']  ==  username, 'start_time'] = updated_dates  
  
def update_entries_time(progress_df): 
    change_start_time('sanjana07', -4, progress_df)
    change_start_time('surinder788', 5, progress_df)
    change_start_time('swaraj87', 1, progress_df)
    change_start_time('tejas2007', 7, progress_df)
    change_start_time('surinder786', 2, progress_df)
    change_start_time('surinder784', 4, progress_df)
    change_start_time('sweth25', -3, progress_df)
    change_start_time('tushar012', -2, progress_df)
    change_start_time('surinder785', -3, progress_df)
    change_start_time('surinder787', 6, progress_df)

def update_file(progress_df): 
    # print(progress_df.columns)
    progress_df.drop(columns = "Unnamed: 0", inplace = True)
    # print(progress_df)
    for i in range(progress_df.shape[0]):
      progress_df['percent_completion'][i] = (progress_df["total_consumption_time"][i]/progress_df["total_media_time"][i])*100
    
    progress_df["expected_time"] = progress_df["total_media_time"].values * 1.25

    x = []
    for i in range(len(progress_df["expected_time"].values)):
      if(progress_df["total_consumption_time"].values[i] == 0 or progress_df["total_consumption_time"].values[i] == np.nan or progress_df["percent_completion"].values[i] == 0):
        x.append(np.nan)
      else:
        x.append(progress_df["expected_time"].values[i]*progress_df["percent_completion"].values[i]*0.01/(progress_df["total_consumption_time"].values[i]))
      

    x = np.array(x)
    x = np.nan_to_num(x)
    y = pd.DataFrame(x)

    progress_df["learning_vel_content"] = x

    progress_df.drop(columns = ["Unnamed: 0.1"], inplace = True)

    start_times = list(progress_df[:]['start_time']) 

    retention_rate = []
    today = datetime.date.today()
    i = 0 
    for d in start_times: 
      if(d != d):
        rate = 0
      else:
        dat = convert_str_to_date(d).date()
        days = (today - dat).days 
        rate = calc_decay_rate(days)
      retention_rate.append(rate)
      i += 1
    progress_df['retention rate'] = retention_rate 

st.set_page_config(page_title="Time Table", layout="wide") 

names = ["aadhirag", "aadil", "aksh", "anand", "ayush", "hansika18", "harshit123", "kartikey123", "mohit08", "piyush222", "prabhat123", "radhakumari79", "sandra55", "sanjana07", "sauravkumar10", "siddhant17", "sneha022", "soorya12", "srijan92", "subh0805", "sumit022", "surinder434", "swaraj87", "sweth25", "tejas2007", "tushar012", "udit1460", "vaibhav31", "vaidik123", "vasu1807", "vishesh1", "yadav101", "yash113", "yogi4565", "yogya2006", "yuktha1212", "yuvraj7890", "zainab99", "zeeshan01", "zubain.c"]
usernames = ["aadhirag2022", "aadil63", "akshverma16", "anand01", "ayushmaurya01", "hansika18", "harshit123", "kartikey123", "mohit08", "piyush222", "prabhat123", "radhakumari79", "sandra55", "sanjana07", "sauravkumar10", "siddhant17", "sneha022", "soorya12", "srijan92", "subh0805", "sumit022", "surinder434", "swaraj87", "sweth25", "tejas2007", "tushar012", "udit1460", "vaibhav31", "vaidik123", "vasu1807", "vishesh1", "yadav101", "yash113", "yogi4565", "yogya2006", "yuktha1212", "yuvraj7890", "zainab99", "zeeshan01", "zubain.c"]

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

credentials = {"usernames":{}}

for un, name, pw in zip(usernames, names, hashed_passwords):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({un:user_dict})

authenticator = stauth.Authenticate(credentials, "app_home", "auth", cookie_expiry_days=0)

name, authentication_status, student_name = authenticator.login("Login", "main")

if authentication_status == False:
  st.error("Username/Password is incorrect")

if authentication_status == None:
  st.warning("Please enter your username and password")

if authentication_status:

  authenticator.logout("Logout", "sidebar")
  st.sidebar.title(f"Welcome {name}")

  @st.cache()
  def read_df():
    progress_df = pd.read_csv('updated_progress_final.csv')
    update_file(progress_df)
    content_df = pd.read_csv('X-Syllabus-content-mapping.csv')
    json_file_path = "taxonomy.json"

    with open(json_file_path, 'r') as j:
      contents = json.loads(j.read())
    
    with open('content_repo.pickle', 'rb') as handle:
      content_repository = pickle.load(handle)

    return progress_df, content_df, contents, content_repository

  def progress():
    progress_df, content_df, contents, content_repository = read_df()
    time=0
    html_temp = """
    <div style="background-image: radial-gradient(circle, rgba(63,94,251,1) 0%, rgba(252,70,107,1) 100%);padding:10px">
    <h2 style="color:white;text-align:center;">Recommendation Engine </h2>
    </div>
    """
    # st.set_page_config(page_title="Time Table", layout="wide") 

    st.markdown(html_temp,unsafe_allow_html=True)
    
    form = st.form(key='my_form')
    chp = form.text_input("Chapter Name")
    form.caption("Enter days")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_selects = [0]
    days_dict= {"Monday":0, "Tuesday":0, "Wednesday":0, "Thursday":0, "Friday":0, "Saturday":0, "Sunday":0}
    col1, col2, col3 = form.columns(3)
    num_hrs = 0
    with col1:
      d0 = st.checkbox(days[0])
      with col1:
        dh0 = st.slider('No. of hours on Monday', 0, 8,None)
        if d0:
          days_dict["Monday"] = dh0
          num_hrs+=dh0
      d1 = st.checkbox(days[1])
      dh1 = st.slider('No. of hours on Tuesday', 0, 8,None)
      if d1:
        days_dict["Tuesday"] = dh1
        num_hrs+=dh1
      d2 = st.checkbox(days[2])
      dh2 = st.slider('No. of hours on Wednesday', 0, 8,None)
      if d2:
        days_dict["Wednesday"] = dh2
        num_hrs+=dh2
    with col2: 
      d3 = st.checkbox(days[3])
      dh3 = st.slider('No. of hours on Thursday', 0, 8,None)
      if d3:
        days_dict["Thursday"] = dh3
        num_hrs+=dh3
      d4 = st.checkbox(days[4])
      dh4 = st.slider('No. of hours on Friday', 0, 8,None)
      if d4:
        days_dict["Friday"] = dh4
        num_hrs+=dh4
      d5 = st.checkbox(days[5])  
      dh5 = st.slider('No. of hours on Saturday', 0, 8,None)
      if d5:
        days_dict["Saturday"] = dh5
        num_hrs+=dh5
    with col3: 
      d6 = st.checkbox(days[6])
      dh6 = st.slider('No. of hours on Sunday', 0, 8,None)
      if d6:
        days_dict["Sunday"] = dh6
        num_hrs+=dh6
    day_selects = [d0, d1, d2, d3, d4, d5, d6]
    days = np.array(days)
    submit_button = form.form_submit_button(label='Get the timetable')
    if submit_button: 
      if (sum(day_selects) != 0):
        table,time, df = get_timetable(content_repository, student_name,chp,progress_df, content_df, contents, days[day_selects], days_dict,num_hrs,sum(day_selects))
      else:
        st.text("Since no days were selected generating a default timetable")
        table, time, df = get_timetable(content_repository, student_name,chp,progress_df,content_df, days_dict, num_hrs,days)
      result=table

      st.markdown("Total time: " + convert(time))
      with st.expander("Time Table", True):
        AgGrid(table, theme='streamlit', height=350, width='100%', fit_columns_on_grid_load=True, autoHeight = True, wrapText = True) 

      st.markdown("Content")
      days = list(df.keys())
      for d in days: 
        with st.expander(d, False):
          twod_arr = df[d]
          for video in twod_arr: 
            [topic, subtopic, time, url] =  video 
            st.markdown(':green[Topic:]     '+  topic)
            st.markdown(':green[Video:]     '+ subtopic)
            st.video(url, format='video/mp4', start_time=0)

        

      st.markdown(" The most viewed content for this chapter:")
      st.text(most_viewed(chp,progress_df))


    if st.button("About"):
      st.markdown(":red[Content and Timetable Recommendation System : ]  IP Project")
      st.text("Suyashi Singhal")
      st.text("Srishti Jain")
      st.text("Gurmehak Kaur")


  def main():
    progress()

  if __name__ == "__main__":
    main() 