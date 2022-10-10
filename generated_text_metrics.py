import spacy

nlp = spacy.load('en_core_web_sm')

def _find_entities(doc):
    ents = []
    if doc.ents:
        for ent in doc.ents:
            ents.append(ent)
    return len(ents), ents


def _noun_information(doc):
    noun_list = []
    for token in doc:
        #PROPN, NOUN
        if token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
            noun_list.append(token.lemma_)
    return len(noun_list), noun_list


def _sentence_information(doc):
    sent_length = []
    repeat_scores = []
    for sent in doc.sents:
        sent_length.append(len(sent))
        token_list = []
        for token in sent:
            token_list.append(token.lemma_)
        score = 1. - len(set(token_list))/len(token_list)
        repeat_scores.append(score)  
    return len(sent_length), sum(sent_length)/len(sent_length), repeat_scores

def generated_text_metric(txt):
    doc = nlp(txt)
    metric = {}
    metric['number_ents'], metric['ents'] = _find_entities(doc)
    metric['number_nouns'], metric['nouns'] = _noun_information(doc)
    metric['number_sents'], metric['avg_sent_len'], metric['sent_repeat_score'] =_sentence_information(doc)
    return metric
    
def generation_sentence_simialrity(txt_prompt, gen_txt):
    gen_doc = nlp(txt_prompt)
    doc = nlp(gen_txt)
    first_sent = True
    similarity = []
    for sent in doc.sents:
        if first_sent:
            first_sent = False
            tmp = sent
            similarity.append(gen_doc.similarity(sent))
        else:
            similarity.append(sent.similarity(tmp))
            tmp = sent
    return similarity

def visualize_dependecy_tree(text):
    doc = nlp(text)
    doc_list = [sent for sent in doc.sents]
    options = {"color": "white", "collapse_phrases" : True, "bg": "#000000"}
    displacy.render(doc_list, style="dep", options=options, jupyter=True)

def visualize_ne_tree(text):
    doc = nlp(text)
    doc_list = [sent for sent in doc.sents]
    displacy.render(doc_list, style="ent", jupyter=True)