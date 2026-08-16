from search import search_keyword


results=search_keyword("prompt injection",top_k=5)

for chunk_id,text,source,path,page,score in results:
    print("Chunk:",chunk_id)
    print("Score:",score)
    print("Source:",source)
    print("Page:",page)
    print(text)
    print("-"*50)