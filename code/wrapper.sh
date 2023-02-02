#full pipeline to process RCT articles

# STEP 1: retrieve abstracts from PubMed using pmid
# pmid.list: one id per line. target article to process; the retrieved abstract file: one section per line

#python retrieve_byPMID.py <pmid.list> <dir to save retrieved abstracts>
python retrieve_byPMID.py data/test/pmid.list data/test/abstracts

# STEP 2: preprocess raw abstracts, tokenize based on sentences and assign section header;
# each abstract will output two files: **.sents and **.tags

#python preprocess_abstracts.py --raw_dir=<dir to retrieved raw abstracts> --sents_dir=<dir to preprocessed abstracts>
python preprocess_abstracts.py --raw_dir=data/test/abstracts --sents_dir=data/test/sents

# STEP 3: MAIN script: run EvidenceMap parser; change parsing parameters in parser_config.py before running.

#python run_parser.py --data_dir=<dir to preprocessed abstracts or raw_abstract> --output_dir=<parsing results>
python run_parser.py --data_dir=data/test/sents --output_dir=data/test/json
