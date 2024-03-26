def pretty_print_docs(docs):
    """
    This helper function is from LangChain docs.
    """
    for ind, doc in enumerate(docs):
        print("-"*100+"\n")
        print(f"Document {ind+1}:\n\n")
        print(doc.page_content)


def get_src_urls(src_docs):
    for d in src_docs:
        # print("1")
        # print(d.metadata["url"])
        print(d.metadata["source"])
