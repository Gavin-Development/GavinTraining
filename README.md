# Gavin Training Repo
This Repo is for the Training of Gavin. A Transformer Based Chat Bot. 
At current time he uses the 
[Reddit Comment Dataset](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/).

### Background
Started back in October 2019 (under a different repo), Gavin became [my (ScotSurvivor)](https://github.com/Scot-Survivor) project outside of studying.
At the time, being only 16 years old, Gavin ended up becoming a lot more than just a project, Gavin became the backbone to my applications towards university,
A-Levels (UK examinations) & even taking an extra a-level to complete in just one year. Despite my age at the time, I still knew just how much work would be
required to achieve the level of coherency & contextual awareness I was aiming for (in fact I still am working towards this!). 

At present day Gavin has come a long way, integrating other papers & modules, even having some C++ modules (written by ShmarvDogg & myself) to speed up certain parts.
Such as, [GavinBackendDatasetUtils](https://github.com/Gavin-Development/GavinBackendDatasetUtils) as well as [GavinTokenizers](https://github.com/Gavin-Development/CPP-SubwordTextEncoder),
which was adapted from the SubwordTextEncoder system that TensorflowDataset uses. 
Furthermore, Gavin now has ties with an incredible discord bot known as [Gerald](https://github.com/Gerald-Development/Barista-Gerald) Written by a  close friend of mine 
[Seb](https://github.com/Elementalmp4) Gerald reaches over 60,000 discord members (as of 26/10/2021), which also means, Gavin is being spoken to by this many members.

### Overall Goals
Gavin's main goal is to be able to speak like your average redditor, while remaining at least some-what humble & polite. This goal is constantly growing in complexity
as I aim for better & better coherency. This is being achieved in several ways, check out the whole [organisation](https://github.com/Gavin-Development) & the repos 
within for the individual goals.

### Specific to this repo
Just to be able to train Gavin, this repo also includes some Dataset tools, which are primarily written for my machine (this is due to [change](https://github.com/Gavin-Development/GavinTraining/tree/BetterDataHandling), 
working on an ob#ject-oriented approach now.)
For specifics on the training script look at main.py.

### Motivations & Inspirations
- [Charles the AI.](https://twitter.com/charles_the_ai?lang=en)
-
- [Sentdex Deep Learning Tutorial.](https://pythonprogramming.net/chatbot-deep-learning-python-tensorflow/)
-
- [Sentdex The Man, the myth and the legend.](https://github.com/Sentdex)
-
- [Ghosty!](https://github.com/TheNitpickyCloud) Well respected Member of sentdex discord server,
  he helps test and develop Gavin in many ways. As well as keeping me motivated to work on Gavin,
  he also helps Give ideas for [Gavin's Discord frontend.](https://github.com/Gavin-Development/GavinDiscordFrontEnd).
-
- [ShmarvDogg.](https://github.com/Shmarvadon) Without him this project wouldn't be possible, 
special thanks to him for lending me his hardware to train and continue to develop Gavin.
-
- [WhoIsAbishag.](https://github.com/WhoIsAbishag) For his work with Irene the ChatBot 
  (AI based on OpenAI). He incites, coherent and contextual responses from Chat Bots.
-
- [Transformer Model.](https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html) Some designs 
are re-made or reforged, this was a key inspiration.
-
- [Performer Paper.](https://arxiv.org/pdf/2009.14794.pdf) Vital to speed improvements during in train.
  
### Build Status
(TODO)

### Framework & Technologies
- [TensorFlow](https://github.com/tensorflow/tensorflow) 
- [TensorFlow Datasets](https://github.com/tensorflow/datasets)
- [Numpy](https://github.com/numpy/numpy)
- [GavinBackend](https://github.com/Gavin-Development/GavinBackend)
- [GavinDatasetUtils](https://github.com/Gavin-Development/GavinBackendDatasetUtils)
- [Cuda](https://developer.nvidia.com/cuda-zone)
- [Cudnn](https://developer.nvidia.com/cudnn)
    
### Features
(TODO)

### Code Examples
(TODO)

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
- "Rethinking attention with performers": 
### Licence
[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.txt). Should have a copy with this software.