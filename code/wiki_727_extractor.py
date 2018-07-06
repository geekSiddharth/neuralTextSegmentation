# from torch.utils.data import Dataset
# from text_manipulation import word_model
# from text_manipulation import extract_sentence_words
import random
import re
from shutil import copy

import wiki_utils
from pathlib2 import Path

section_delimiter = "========"


def get_random_files(count, input_folder, output_folder, specific_section=True):
    files = Path(input_folder).glob('*/*/*/*') if specific_section else Path(input_folder).glob('*/*/*/*/*')
    file_paths = []
    for f in files:
        file_paths.append(f)

    random_paths = random.sample(file_paths, count)

    for random_path in random_paths:
        output_path = Path(output_folder).joinpath(random_path.name)
        copy(str(random_path), str(output_path))


def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files


def get_cache_path(wiki_folder):
    cache_file_path = wiki_folder / 'paths_cache'
    return cache_file_path


def cache_wiki_filenames(wiki_folder):
    files = Path(wiki_folder).glob('*/*/*/*')
    cache_file_path = get_cache_path(wiki_folder)

    with cache_file_path.open('w') as f:
        for file in files:
            f.write(unicode(file) + u'\n')


def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section


def get_scections_from_text(txt, high_granularity=True):
    sections_to_keep_pattern = wiki_utils.get_seperator_foramt() if high_granularity else wiki_utils.get_seperator_foramt(
        (1, 2))
    if not high_granularity:
        # if low granularity required we should flatten segments within segemnt level 2
        pattern_to_ommit = wiki_utils.get_seperator_foramt((3, 999))
        txt = re.sub(pattern_to_ommit, "", txt)

        # delete empty lines after re.sub()
        sentences = [s for s in txt.strip().split("\n") if len(s) > 0 and s != "\n"]
        txt = '\n'.join(sentences).strip('\n')

    all_sections = re.split(sections_to_keep_pattern, txt)
    non_empty_sections = [s for s in all_sections if len(s) > 0]

    return non_empty_sections


def get_sections(path, high_granularity=True):
    file = open(str(path), "r")
    raw_content = file.read()
    file.close()

    clean_txt = raw_content.decode('utf-8').strip()

    sections = [clean_section(s) for s in get_scections_from_text(clean_txt, high_granularity)]
    required_sections = sections[1:]
    required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]
    return required_non_empty_sections
