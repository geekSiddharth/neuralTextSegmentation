# Data Processing
- See process_documents.py to change parameters of processing data

# Adding a new dataset
- See load_documents.py and add code to it

## Data Format

### moderated videos segments 

This folder contains a lot of json files with name is the format of `<youtube_video_id>.json`. Each file contains a `segments`.

### ted

This folder contains a single file. That file contains a `dict`/`json` of `segments`.
Format:
```json
    {
      "ted_talk_id":"segments(as given below)"
    }
```


### udacity

This folder contains a single file. That file contains a list of doc in the following format
```
[[doc1],[doc2],[doc3],[doc4],[doc5]....]
doc1 = ["segment1 text", "segment2 text", "segment3 text"]
```

#### What is a `segments` ?

`segments` is a dictionary or a json of the following format:
```json
    {
     "1":  {
            "cantidate_title": "",
            "end_time": "",
            "index": "",
            "start_time": "",
            "summary": "",
            "text": "",
            "title": ""
        },
     "2": {
            "cantidate_title": "",
            "end_time": "",
            "index": "",
            "start_time": "",
            "summary": "",
            "text": "",
            "title": ""
        }
    } 
```

- `segments[i]` will give you the i_th segment in the doc/transcript(indexing starts from 1). 
- `segments[i]['index']` will most probably be equal to `i`. 
- `segments[i]['text']` will give textual content of that segment in form of a string
- `segments[i]['title']` will give you the title of the segment as per the ToC
- `segments[i]['start_time']` will give you the starting time of the segment as per the ToC.( in the form of ['hr', 'min', 'sec'])
