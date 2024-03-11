# def get_paragraphs(from_item):
#     extracted_text = ""
#     paragraphs = from_item.find_all("p")
#     for paragraph in paragraphs:
#         if extracted_text == "":
#             extracted_text = paragraph.text
#         else:
#             extracted_text = extracted_text + "\n" + paragraph.text

#     return extracted_text

def get_paragraphs(from_item):
    extracted_text = ""
    paragraphs = from_item.find_all(["title", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p"])
    for paragraph in paragraphs:
        if extracted_text == "":
            extracted_text = paragraph.text
        else:
            extracted_text = extracted_text + "<annotate_paragraph>" + paragraph.text

    return extracted_text