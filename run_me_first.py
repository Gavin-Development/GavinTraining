import os
import platform
import time
import nltk
print("Setting up...")
if not os.path.exists('./bunchOfLogs'):
    os.mkdir('./bunchOfLogs')
if not os.path.exists("./temp"):
    os.mkdir('./temp')
os.system("python -m pip install -r requirements.txt")
print("Installing Stop words from NTLK")
nltk.download(["stopwords"])
if platform.system() == "Linux":
    print("Detected Linux. Attempting to install graphviz")
    os.system("sudo apt-get install graphviz")
elif platform.system() == "Windows":
    print("Downloading Graphviz for Windows.")
    os.system("curl https://gitlab.com/graphviz/graphviz/-/package_files/7097345/download -o ./temp/Graphviz.exe")
    print("This script will automatically run the installer for Graphviz 64bit. In 3 seconds")
    time.sleep(3)
    os.system("cd temp/ && Graphviz")
print("""Please go to: https://drive.google.com/file/d/1o0WrKqDQGOcA0xtQ1eubHZJ9B18YYxL8/view
Download the file and extract its contents into the same place as RUNTHIS-old.py (Or you can changed the environment var inside
main-old.py/RUNTHIS-old.py to where you've extracted the files.""")
time.sleep(10)
print("Cleaning up")
os.remove('temp')
print("Finished!")
time.sleep(.2)
