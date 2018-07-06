"""
This file contains functions that loads raw data from the disk and formats them to the desired method for further processing
If you want to load a new dataset, this is where you will adding a function and that function should be added to load_all()
TODO: See function _load_new_dataset()
TODO: See function load_all()

Other processing files will be using load_all() function only
"""

import glob
import json

import nltk

nltk.download('punkt')


def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def _load_ted():
    documents = []

    # TED A lot them skips with avg seg len 10 and around 500 with seg len 5
    print("Stating to Load TED docs")

    with open('data/ted/ted_segments.txt') as f:
        ted_data = json.load(f)

    wuuutt = False  # To combine stuff with ()

    for doc_id in ted_data.keys():
        sections = []
        for segment_id in ted_data[doc_id].keys():
            text = remove_non_ascii(ted_data[doc_id][segment_id]['text']).strip()
            if text.startswith("(") and text.endswith(")"):
                if len(sections) > 0:
                    wuuutt = True
                continue

            lines = nltk.sent_tokenize(text)

            if wuuutt:
                # previous section was of format (laugh) or (some text)
                try:
                    # combines the current section with the previous sections
                    sections[len(sections) - 1].extend(lines)
                except:
                    pass
                wuuutt = False
            else:
                # normal stuff
                sections.append(lines)
        documents.append(("ted" + str(doc_id), sections))
    print("Done Load TED docs")

    return documents


def _load_udacity():
    documents = []

    # Udacity around //200 skips with avg seg len 5
    print("Stating to Load Udacity docs")
    with open('data/udacity/udacity.txt') as f:
        udacity = json.load(f)

    counter = 0
    for doc in udacity:
        sections = []
        for segment in doc:
            lines = nltk.sent_tokenize(remove_non_ascii(segment))
            sections.append(lines)
        documents.append(("udacity" + str(counter), sections))
        counter += 1
    print("Done Loading Udacity docs")
    return documents


def _load_moderated_videos():
    documents = []
    # Moderated VIDEOS //around 300 valid docs
    counter = 0
    print("Starting to load moderated videos docs")
    for file_doc in sorted(glob.glob('data/moderated_videos_segments/*.json')):

        # Reading content of  a particular .json file
        with open(file_doc) as f:
            doc_ = json.load(f)

        sections = []
        for segment_id in doc_.keys():
            lines = nltk.sent_tokenize(remove_non_ascii(doc_[segment_id]['text']))
            sections.append(lines)

        documents.append(("moderated_vid" + file_doc, sections))
        counter += 1
    print("Done loading moderated videos docs")
    return documents


def _load_new_dataset():
    documents = []

    """
        How the document needs to be encoded
        documents: [("ted12", [ [line1,line2..],
                                   [line1,line2..]
                                 ]
                        ),
                        ("ted13",.. )
                       ]
        It is a list of tuples. (document_id, segments)
        segments is a list of segment
        segment is a list of lines in that segment
    """
    for doc in documents:
        for section in doc:
            for line in section:
                print(line)
                pass
        documents.append(("docID", doc))

    return documents


def load_all():
    documents = []

    documents.extend(_load_ted())
    documents.extend(_load_udacity())
    documents.extend(_load_moderated_videos())

    # Loading a new data_set
    # documents.extend(_load_new_dataset())

    return documents
