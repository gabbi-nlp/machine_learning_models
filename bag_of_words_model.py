#bag_of_words_model.py
import string

"""
a count for each word in a sentence

In this article, we are going to discuss a Natural Language Processing technique of 
text modeling known as Bag of Words model. Whenever we apply any algorithm in NLP, 
it works on numbers. We cannot directly feed our text into that algorithm. Hence, Bag of Words model 
is used to preprocess the text by converting it into a bag of words, which keeps a count 
of the total occurrences of most frequently used words

"""
def preprocessing(docs):
    # creating one entire string
    docs = ' '.join(docs)

    # removing punctuation and normalizing string
    remove_punc = str.maketrans('', '', string.punctuation)
    cleaned_docs = docs.translate(remove_punc)
    cleaned_docs = cleaned_docs.lower()
    cleaned_docs = cleaned_docs.strip()
    cleaned_docs = cleaned_docs.split()

    # gets list of unique values
    doc_vocabulary = list(set(cleaned_docs))

    return doc_vocabulary

def get_sent_dictionary(sent):
    sent = sent.lower()
    sent_word_count = {}
    
    # counts the number of words
    for word in sent.split():
        if word in sent_word_count.keys():
            sent_word_count[word] += 1
        else:
            sent_word_count[word] = 1

    return sent_word_count

# builds a language dictionary based on the input doc and 
# generates embeddings for each sentence in doc
def bag_of_words_lm(docs):
    embeddings = []
    docs = docs.split('.')
    doc_vocabulary = preprocessing(docs)
    
    # genearte embedding per sentence
    for sentence in docs:
        sent_embed = []
        sent_word_count = get_sent_dictionary(sentence)

        for word in doc_vocabulary:
            # get numeric occurence of word
            word_count_value = sent_word_count[word] if word in sent_word_count.keys() else 0

            # place value at the index associated with the word in vector: sent_embed
            sent_embed.append(word_count_value)
        
        # add to all sentence embeddings
        embeddings.append(sent_embed)

    print(embeddings)


if __name__ == "__main__":
    # pass in a sentences
    docs = input("input docs: ")
    
    # call BoW language model
    bag_of_words_lm(docs)