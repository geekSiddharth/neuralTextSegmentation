import sys

from load_documents import load_all
from process_line import LineProcessor

sys.path.append("../")

MIN_SENTENCES_IN_DOCUMENT = 21  # currently equal to 2*context + 1
MIN_SENTENCES_IN_SECTION = 1
MIN_SECTIONS = 2
SENTENCE_LENGTH = 20  # Number of words in a sentence


def _filter_documents(documents):
    """
    Removes bad documents from the given documents which doesn't satisfy
        @:param MIN_SECTIONS
        @:param MIN_SENTENCES_IN_DOCUMENT
        @:param MIN_SENTENCES_IN_SECTION

    :param documents: [("ted12", [ [line1,line2..],
                                   [line1,line2..]
                                 ]
                        ),
                        ("ted13",.. )
                       ]
        It is a list of tuples. (document_id, segments)
        segments is a list of segment
        segment is a list of lines in that segment
    :return: `documents` with same above structure but with bad documents removed
    """
    print("Filtering GOOD documents using MIN_(SENT/DOC/SEC..) filters ....")
    best_docs = []
    for (docID, sections) in documents:

        # Remove documents with less than 2 sections
        if MIN_SECTIONS != -1:
            if len(sections) <= MIN_SECTIONS:
                print(docID, ": Fails at MIN_SECTIONS (", len(sections), "/", MIN_SECTIONS, ")")
                continue

        sentence_counts = [[len(par) for par in section] for section in sections]

        # Remove documents that have less than MIN_SENTENCES_IN_DOCUMENT.
        if MIN_SENTENCES_IN_DOCUMENT != -1:
            count = sum([sum(section) for section in sentence_counts])
            if count < MIN_SENTENCES_IN_DOCUMENT:
                print(docID, ": Fails at MIN_SENTENCES_IN_DOCUMENT (", count, "/", MIN_SENTENCES_IN_DOCUMENT, ")")
                continue

        # Remove documents that have less than MIN_SENTENCES_IN_SECTION
        if MIN_SENTENCES_IN_SECTION != -1:
            count = min([sum(section) for section in sentence_counts])
            if count < MIN_SENTENCES_IN_SECTION:
                print(docID, ": Fails at MIN_SENTENCES_IN_SECTION (", count, "/", MIN_SENTENCES_IN_SECTION, ")")
                continue

        best_docs.append(docID)

    best_docs = set(best_docs)
    new_docs = [doc for doc in documents if doc[0] in best_docs]
    return new_docs


def create_doc_sentence_sequence(documents):
    """
    This converts documents to
    X -> [ doc1_X, doc2_X, doc3_X ...] where doc_i_X = [line1, line2, line3..]
    Y -> [doc1_Y, doc2_Y, doc3_Y ...] where doc_i_Y = [True, False, False ..]
    True implies that the corresponding line is a start of segment

    line-i is padded and processed by process_line.py

    :param documents: as used everywhere else
    :return: X,Y as described above
    """
    sequence_x = []
    sequence_y = []

    line_processor = LineProcessor()

    for docID, doc in documents:
        segments = []
        labels = []
        for section_id in range(len(doc)):
            for line_id in range(len(doc[section_id])):
                label = False
                if line_id is 0:
                    label = True

                line = doc[section_id][line_id]

                """
                    WORD/TEXT PROCESSING
                """
                line = line_processor.fit(line, SENTENCE_LENGTH)

                segments.append(line)
                labels.append(label)

        sequence_x.append(segments)
        sequence_y.append(labels)

    return sequence_x, sequence_y


def load_dataset():
    """
    X -> [ doc1_X, doc2_X, doc3_X ...] where doc_i_X = [line1, line2, line3..]
    Y -> [doc1_Y, doc2_Y, doc3_Y ...] where doc_i_Y = [True, False, False ..]
    True implies that the corresponding line is a start of segment

    line-i is padded and processed by process_line.py (it has a list of indexes)
    :return: X,Y
    """
    print("Loading doc files")
    documents = load_all()
    documents = _filter_documents(documents)
    print("Starting to transform to indexes")
    X, Y = create_doc_sentence_sequence(documents)

    return X, Y
