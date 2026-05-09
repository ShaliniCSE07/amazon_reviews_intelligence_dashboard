import yake

def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor(top=10)
    keywords = kw_extractor.extract_keywords(text)

    result = []
    for keyword, score in keywords:
        result.append(keyword)

    return result