import _random
from random import Random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait


class CustomRandom(Random):
    def shuffle(self, x, random=None):
        """Shuffle list x in place, and return None.
                Optional argument random is a 0-argument function returning a
                random float in [0.0, 1.0); if it is the default None, the
                standard random.random will be used.
                """

        if random is None:
            randbelow = self._randbelow
            for i in tqdm(reversed(range(1, len(x))), desc="Shuffle", total=len(x)):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = randbelow(i + 1)
                x[i], x[j] = x[j], x[i]
        else:
            _int = int
            for i in tqdm(reversed(range(1, len(x))), desc="Shuffle", total=len(x)):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = _int(random() * (i + 1))
                x[i], x[j] = x[j], x[i]


def readFile(file):
    with open(f"D:\\Datasets\\reddit_data\\files\\{file}", "r", encoding="utf8") as file:
        lines = file.readlines()
    return lines


def writeFile(file, data):
    with open(f"D:\\Datasets\\reddit_data\\files\\{file}", "w", encoding="utf8") as f:
        for i in tqdm(range(len(data)), desc=f"{file}", total=len(data)):
            f.write(data[i])
        f.close()
    return True


def writeFiles(file1, data1, file2, data2):
    with ThreadPoolExecutor(2) as executor:
        fut1 = executor.submit(writeFile, file1, data1)
        fut2 = executor.submit(writeFile, file2, data2)
        wait((fut1, fut2))

        exc1 = fut1.exception()
        if exc1 is not None:
            raise exc1

        exc2 = fut2.exception()
        if exc2 is not None:
            raise exc2
        return fut1.result(), fut2.result()


def readFiles(file1, file2):
    with ThreadPoolExecutor(2) as executor:
        fut1 = executor.submit(readFile, file1)
        fut2 = executor.submit(readFile, file2)
        wait((fut1, fut2))

        exc1 = fut1.exception()
        if exc1 is not None:
            raise exc1

        exc2 = fut2.exception()
        if exc2 is not None:
            raise exc2
        return fut1.result(), fut2.result()


def chunks(input_list, size_chunk):
    for i in range(0, len(input_list), size_chunk):
        yield input_list[i:i + size_chunk]


def shuffleFiles():
    C = CustomRandom(_random.Random)
    print("Reading files...")
    lines, lines2 = readFiles("train.from", "train.to")
    print("Finished reading files...")
    print("Zipping...")
    shuffleThis = list(zip(lines, lines2))
    print("Done zipping")
    print("Shuffling...")
    C.shuffle(shuffleThis)
    print("Done shuffling...")
    print("Unzipping...")
    lines, lines2 = zip(*shuffleThis)
    print("Unzipped")
    print("Beginning write")
    check1, check2 = writeFiles("train.from", lines, "train.to", lines2)
    if check1 and check2:
        print("All processes finished")
    else:
        print(f"First File completed: {check1}\nSecond File completed: {check2}")


shuffleFiles()