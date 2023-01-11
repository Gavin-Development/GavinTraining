import sys
import tqdm
sys.path.append('GavinBackend/CustomPackages/windows/')
import GavinBackendDatasetUtils as LTD


def text_generator(chunk=100):
    file_handle = open("D:\\Datasets\\reddit_data\\files\\plaintext.txt", "r")
    for sample in file_handle:
        if sample.strip() == "":
            continue
        else:
            yield sample.strip().split(" ")


if __name__ == "__main__":
    Production_Tokenizer = LTD.Tokenizer()
    done = False
    gen = text_generator()
    pbar = tqdm.tqdm(total=287738417)
    while not done:
        try:
            line = next(gen)
            if len(line) > 0:
                Production_Tokenizer.BuildEncodes_GPU(line)
            pbar.update(len(line))
        except StopIteration:
            done = True
    pbar.close()
    Production_Tokenizer.Save()
