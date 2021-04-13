import os
import re
import tensorflow_datasets as tfds

path_to_dataset = "cornell movie-dialogs corpus"

path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
path_to_movie_conversations = os.path.join(path_to_dataset, "movie_conversations.txt")


def preprocess_sentence(sentence):
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    pattern_a = re.compile(r"([?.!,'])")
    pattern_b = re.compile(r"[^a-zA-Z?.!,']+")
    sentence = re.sub(pattern_a, r"\1", sentence)
    sentence = re.sub(pattern_b, " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = sentence.strip()
    # adding start and an end token to the sentence
    return sentence


# noinspection PyShadowingNames,PyPep8Naming
def load_conversations(max_len):
    with open("D:\\Datasets\\reddit_data\\files\\train.from", "r", encoding="utf8", buffering=1000) as file:
        newline = " newlinechar "
        for line in file:
            if newline in line:
                line = line.replace(newline, "\n")
            if len(line) <= max_len:
                yield line
        file.close()

    with open("D:\\Datasets\\reddit_data\\files\\train.to", "r", encoding="utf8", buffering=1000) as file:
        newline = " newlinechar "
        for line in file:
            if newline in line:
                line = line.replace(newline, "\n")
            if len(line) <= max_len:
                yield line
        file.close()


if __name__ == "__main__":
    MAX_LENGTH = 40 + 2
    TARGET_VOCAB_SIZE = 2**20  # 1048576
    save_path = "Tokenizer-1Million"
    movie_dialog_max = 0

    gen = load_conversations(MAX_LENGTH)

    print("Starting Tokenizer this may take a while....")
    # Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        gen, target_vocab_size=TARGET_VOCAB_SIZE, max_subword_length=12)
    tokenizer.save_to_file(f"{save_path}")
