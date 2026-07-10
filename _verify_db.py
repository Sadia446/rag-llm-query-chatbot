"""Verify that the Chroma DB can be loaded and queried."""
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

db = Chroma(persist_directory="chroma", embedding_function=get_embedding_function())
items = db.get(include=[])
print(f"Documents in DB: {len(items['ids'])}")

results = db.similarity_search_with_score("monopoly", k=3)
print(f"Search returned {len(results)} results")
for doc, score in results:
    print(f"  - score={score:.4f}  id={doc.metadata.get('id','?')}")

print("\nDB verification PASSED")
