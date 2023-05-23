import pickle
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth

names = ["aadhirag", "aadil", "aksh", "anand", "ayush", "hansika18", "harshit123", "kartikey123", "mohit08", "piyush222", "prabhat123", "radhakumari79", "sandra55", "sanjana07", "sauravkumar10", "siddhant17", "sneha022", "soorya12", "srijan92", "subh0805", "sumit022", "surinder434", "swaraj87", "sweth25", "tejas2007", "tushar012", "udit1460", "vaibhav31", "vaidik123", "vasu1807", "vishesh1", "yadav101", "yash113", "yogi4565", "yogya2006", "yuktha1212", "yuvraj7890", "zainab99", "zeeshan01", "zubain.c"]
usernames = ["aadhirag2022", "aadil63", "akshverma16", "anand01", "ayushmaurya01", "hansika18", "harshit123", "kartikey123", "mohit08", "piyush222", "prabhat123", "radhakumari79", "sandra55", "sanjana07", "sauravkumar10", "siddhant17", "sneha022", "soorya12", "srijan92", "subh0805", "sumit022", "surinder434", "swaraj87", "sweth25", "tejas2007", "tushar012", "udit1460", "vaibhav31", "vaidik123", "vasu1807", "vishesh1", "yadav101", "yash113", "yogi4565", "yogya2006", "yuktha1212", "yuvraj7890", "zainab99", "zeeshan01", "zubain.c"]
passwords = ["a1234", "a1235", "a1236", "a1237", "a1238", "a1239", "a1240", "a1241", "a1242", "a1243", "a1244", "a1245", "a1246", "a1247", "a1248", "a1249", "a1250", "a1251", "a1252", "a1253", "a1254", "a1255", "a1256", "a1257", "a1258", "a1259", "a1260", "a1261", "a1262", "a1263", "a1264", "a1265", "a1266", "a1267", "a1268", "a1269", "a1270", "a1271", "a1272", "a1273"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"

with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)