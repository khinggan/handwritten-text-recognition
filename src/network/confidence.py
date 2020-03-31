import editdistance
from operator import itemgetter


def lexicon_based_confidence(predicts: list):
    with open("C:/work/handwritten-text-recognition/raw/iam_words/words.txt", 'r') as f:
        lines = [line.strip() for line in f.readlines() if line[0] != '#']
        words = [line.split(" ")[-1].lower() for line in lines if line.split(" ")[1] == "ok"]
    word_count = dict()
    for item in words:
        word_count[item] = word_count.get(item, 0) + 1

    word_set = set(words)
    final_predicts, confidences = list(), list()
    for word in predicts:
        f_predict, confidence = lexicon_based_word_confidence(word_set, word_count, word)
        final_predicts.append(f_predict)
        confidences.append(confidence)
    return final_predicts, confidences


def lexicon_based_word_confidence(word_set, word_count, test_word: str) -> (str, float):
    """
    editdistance == 0 ----> confidence = 1, candidate = test word
    editdistance != 0
        sort by editdistance:
        first min editdistance == second min editdistance:
             get first large freq of min editdistance and second large freq of min editdistance
             1-freq2/freq1
        else:
            constant
    :param test_word: calculated confidence word
    """

    test_word = test_word.lower()
    candidate, confidence = "", 0.0
    candidates = []
    for word in word_set:
        if editdistance.distance(test_word, word) == 0:
            candidate = word
            confidence = 1.0
            break
        elif editdistance.distance(test_word, word) <= 2:
            candidates.append((word, editdistance.distance(test_word, word), word_count[word]))
    if candidate == "":
        candidates.sort(key=itemgetter(1))
        if len(candidates) >= 2:
            if candidates[0][1] == candidates[1][1]:
                first_max_freq, second_max_freq = 0, 0
                for item in candidates:
                    if item[1] == candidates[0][1]:
                        if item[2] > first_max_freq:
                            first_max_freq = item[2]
                            candidate = item[0]
                for item in candidates:
                    if item[1] == candidates[0][1]:
                        if second_max_freq < item[2] < first_max_freq:
                            second_max_freq = item[2]
                confidence = 1 - second_max_freq / first_max_freq
            else:
                candidate = candidates[0][0]
                confidence = 0.4
        else:   # candidates length is 0 or 1
            candidate = candidates[0][0] if len(candidates) == 1 else ""
            confidence = 1.0
    return candidate, confidence
