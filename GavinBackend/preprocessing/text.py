import re


def preprocess_sentence(sentence):
    sentence = sentence.strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,'*\":@])", r" \1 ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    sentence = re.sub(r"[^a-zA-z?.!,'*:\"@]+", " ", sentence)
    sentence = sentence.strip()
    # adding start and an end token to the sentence
    return sentence


def preprocess_context(sentence):
    sentence = re.sub(r"[^a-zA-Z ]+", "", sentence)
    return sentence


# noinspection PyShadowingNames,PyPep8Naming
def load_data(reddit_set_max, movie_dialog_max, path_to_movie_lines, path_to_movie_conversations):
    id2line = {}
    inputs, outputs = [], []
    reddit_line = 0
    if movie_dialog_max > 0:
        with open(path_to_movie_lines, errors="ignore") as file:
            lines = file.readlines()
        for line in lines:
            parts = line.replace('\n', '').split(' +++$+++ ')
            id2line[parts[0]] = parts[4]

        with open(path_to_movie_conversations, 'r') as file:
            lines2 = file.readlines()
        for line2 in lines2:
            parts = line2.replace('\n', '').split(" +++$+++ ")
            # get the conversation in a list of line ID
            conversation = [line2[1:-1] for line2 in parts[3][1:-1].split(', ')]
            for i in range(len(conversation) - 1):
                outputs.append(preprocess_sentence(id2line[conversation[i]]))
                inputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
                if len(inputs) >= movie_dialog_max:
                    break
    if reddit_set_max + movie_dialog_max >= 1_000_000:
        with open("D:\\Datasets\\Humour\\humourQ.txt", "r") as f:
            for line in f:
                inputs.append(preprocess_sentence(line))
                reddit_set_max -= 1
                reddit_line += 1
        with open(f"D:\\Datasets\\Humour\\humourA.txt", "r") as f:
            for line in f:
                outputs.append(preprocess_sentence(line.capitalize()))
                reddit_set_max -= 1
                reddit_line += 1

    with open("D:\\Datasets\\reddit_data\\files\\train.from", "r", encoding="utf8", buffering=1000) as file:
        newline = " newlinechar "
        for line in file:
            if newline in line:
                line = line.replace(newline, "\n")
            inputs.append(line)
            if len(inputs) >= reddit_set_max / 2:
                break
        file.close()

    with open("D:\\Datasets\\reddit_data\\files\\train.to", "r", encoding="utf8", buffering=1000) as file:
        newline = " newlinechar "
        for line in file:
            if newline in line:
                line = line.replace(newline, "\n")
            outputs.append(line)
            if len(outputs) >= reddit_set_max / 2:
                file.close()
                return inputs, outputs
        file.close()
    return inputs, outputs

