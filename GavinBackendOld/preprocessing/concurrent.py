import marshal
from GavinBackendOld.preprocessing.text import preprocess_sentence
from collections import Iterable
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
from typing import List, Union, IO, AnyStr


def preprocess_process(sentences: list) -> list:
    outputs = []
    for sentence in sentences:
        outputs.append(preprocess_sentence(sentence))
    return outputs


def chunk(lst: list, count: int) -> List:  # Make a list into N number of lists
    size = len(lst) // count  # figure out the size of them all
    for i in range(0, count):
        s = slice(i * size, None if i == count - 1 else (
                                                                i + 1) * size)  # Using slice here because you can't store 2:3 as a variable
        yield lst[s]  # Yield the list


def flatten(lis: Union[List[List] or list]) -> Union[List[List] or list]:
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for i in flatten(item):
                yield i
        else:
            yield item


def save_marshal(item: object, f_path: str) -> None:
    file = open(f_path, 'ab')
    marshal.dump(item, file)
    file.close()


def save_files(item1: object, item2: object, file1: IO[AnyStr], file2: IO[AnyStr]):
    with ThreadPoolExecutor(2) as executor:
        executor.submit(save_marshal, item1, file1)
        executor.submit(save_marshal, item2, file2)
        wait((executor.submit(save_marshal, item1, file1), executor.submit(save_marshal, item2, file2)))


def regex_main(questions: List[AnyStr], answers: List[AnyStr], regex_cores: int):
    print("Starting Regex....")
    # Create the Generator Objects
    generator_q = chunk(questions, regex_cores)
    generator_a = chunk(answers, regex_cores)

    # Use a 2D List to split the questions/answers into chunks for each core (How many used defined by Regex_cores)

    with ProcessPoolExecutor(max_workers=regex_cores) as executor:
        process_outputs = executor.map(preprocess_process, generator_q)
        questions = list(flatten(process_outputs))
        executor.shutdown()

    with ProcessPoolExecutor(max_workers=regex_cores) as executor:
        process_outputs = executor.map(preprocess_process, generator_a)
        answers = list(flatten(process_outputs))
        executor.shutdown()

    print("Finished Regex")
    return questions, answers

