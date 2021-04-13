# Gavin Training Repo
This Repo is for the Training of Gavin. A Transformer Based Chat Bot. 
At current time he uses the 
[Reddit Comment Data](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/).
This Allows for some rather coherent responses. Under the correct Training conditions. 

### Motivations & Inspirations
- [Charles the AI](https://twitter.com/charles_the_ai?lang=en)
-
- [Sentdex Deep Learning Tutorial](https://pythonprogramming.net/chatbot-deep-learning-python-tensorflow/)
-
- [Sentdex The Myth and The Legend](https://github.com/Sentdex)
-
- [Ghosty!](https://github.com/TheNitpickyCloud) Well respected Member of sentdex discord server,
  he helps test and develop Gavin in many ways. As well as keeping me motivated to work on Gavin,
  he also helps Give ideas for [Gavin's Discord frontend](https://github.com/Scot-Survivor/GavinDiscordFrontEnd).
-
- [ShmarvDogg](https://github.com/Shmarvadon) Without him this project wouldn't be possible, 
special thanks to him for lending me his hardware to train and continue to develop Gavin.
-
- [WhoIsAbihag](https://github.com/WhoIsAbishag) For his work with Irene the ChatBot 
  (AI based on OpenAI). He incites, coherent and contextual responses from Chat Bots.
-
- [Transformer Model](https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html) Some designs 
are re-made or reforged, this was a key inspiration.
-
- [Reformer](https://openreview.net/pdf?id=rkgNKkHtvB) Code adapted from here.
  
### Build Status
(TODO)

### Framework & Technologies
1. Built with:
   - [TensorFlow](https://github.com/tensorflow/tensorflow)
   - [TensorFlow Datasets](https://github.com/tensorflow/datasets)
   - [Numpy](https://github.com/numpy/numpy)
    
2. Training:
   - [Cuda](https://developer.nvidia.com/cuda-zone)
   - [Cudnn](https://developer.nvidia.com/cudnn)
    
### Features
(TODO)

### Code Examples
(TODO)

### Installation
If you're running this for Scot-Survivor then please use the [RUNTHIS.py](https://github.com/Scot-Survivor/GavinTraining/blob/master/RUNTHIS.py)
This is preconfigured with settings that Scot-Survivor needs. 
- Setup
  - Download the dataset [files.7z](https://drive.google.com/drive/folders/1GDlTigX4x-H4F7SSqg3QPf3A-byJ9N-v?usp=sharing)
  - Extract the dataset
  - Set the environment variable `REDDIT_DATASET_PATH` to location of extraction inside [RUNTHIS.py](https://github.com/Scot-Survivor/GavinTraining/blob/master/RUNTHIS.py) or [main.py](https://github.com/Scot-Survivor/GavinTraining/blob/master/main.py)
  - If [main.py](https://github.com/Scot-Survivor/GavinTraining/blob/master/main.py)
    - Follow the on-screen prompts and type in your settings
    - Wait.
  - If [RUNTHIS.py](https://github.com/Scot-Survivor/GavinTraining/blob/master/RUNTHIS.py)
    - The script is already pre-configured you're good to go! 
    

### Tests
- ![Input: you're dumb Output: * laughs * I don ' t get it .](https://github.com/Scot-Survivor/GavinTraining/blob/master/funny-responses/image1.PNG?raw=true)
- ![Input: hi Output: * kisses back * Input: kiss Output: * laughs * I love you too .](https://github.com/Scot-Survivor/GavinTraining/blob/master/funny-responses/image2.png?raw=true)
- ![Input: what is the point of life? Output: I think it was a joke .](https://github.com/Scot-Survivor/GavinTraining/blob/master/funny-responses/image3.png?raw=true)
- ![Input: get smart Output: I don ' t think so .](https://github.com/Scot-Survivor/GavinTraining/blob/master/funny-responses/image4.png?raw=true)


### Contribute
(TODO)

### Credits
- "REFORMER: THE EFFICIENT TRANSFORMER": https://arxiv.org/pdf/2001.04451.pdf
- "Practical and Optimal LSH for Angular Distance": https://arxiv.org/pdf/1509.02897.pdf
- "When and Why is Document-level Context Useful in Neural Machine Translation?": https://arxiv.org/pdf/1910.00294.pdf
- "Attention Is All You Need": https://arxiv.org/pdf/1706.03762.pdf
- "Improved Transformer Architecture for Sequence to Sequence Translation": https://www.cs.princeton.edu/sites/default/files/austin_wang_spring_2019.pdf- "Combining Local and Document-Level Context: The LMU Munich Neural Machine Translation System at WMT19": https://www.aclweb.org/anthology/W19-5345.pdf
- "Improving the Transformer Translation Model with Document-Level Context": https://arxiv.org/pdf/1810.03581.pdf
- "Illustrating The Reformer": https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0
- "Reformer: The Efficient Transformer: https://openreview.net/pdf?id=rkgNKkHtvB"
### Licence
(TODO)