from langchain.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="mistral", base_url="http://localhost:11434")
vector = embeddings.embed_query("тест")
print(len(vector))