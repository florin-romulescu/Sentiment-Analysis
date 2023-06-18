from os.path import exists

MODEL_PATH = "Model.sav"

if __name__ == "__main__":
    if not exists(MODEL_PATH):
        print("Run `make generate` to create the model.")
        print("Exiting...")
    else:
        import pickle
        from nltk.tokenize import word_tokenize

        model = pickle.load(open(MODEL_PATH, "rb"))

        while True:
            sentence = input('Input: ')
            if sentence == 'exit': break
            sentence = word_tokenize(sentence)
            print("Sentiment: ", model.classify(dict([token, True] for token in sentence)))