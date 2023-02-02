import codecs, re, sys, os,warnings
warnings.filterwarnings("ignore")
import spacy
nlp = spacy.load("en_core_sci_lg")
import tensorflow as tf
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "raw_dir", None,
    "The input abstract datadir.")

flags.DEFINE_string(
    "sents_dir", None,
    "The output tokenized sents datadir.")

CONSORT_tags=["TITLE","BACKGROUND","PURPOSE","METHODS","RESULTS","CONCLUSION"]
def preprocess_ab(raw_section):

    match = re.search("^[A-Z]+ : ",raw_section)
    if match:
        info = raw_section.split(" : ")
        if info[0].isupper() :
            tag = info[0]
            raw_section = " ".join(info[1:])
        else:
            tag = 'UNKNOWN'
            raw_section = raw_section
        doc = nlp(raw_section.rstrip())            
        sents = [sent.text for sent in doc.sents]
        return tag,sents
    else:
        doc = nlp(raw_section.rstrip())
        tag = 'UNKNOWN'
        sents = [sent.text for sent in doc.sents]
        return tag,sents



def main():
    if not os.path.exists(FLAGS.sents_dir):
        try:
            #createdir= "mkdir "+FLAGS.sents_dir
            os.mkdir(FLAGS.sents_dir)
        except:
            print("DIR ERROR! Unable to create this directory!")
    exception = codecs.open(os.path.join(FLAGS.sents_dir, "exception.ids"),"w")
    for f in os.listdir(FLAGS.raw_dir):
        
        try:
            if not re.search("\.txt",f):
                continue
            pmid = re.sub("\.txt","",f)
            infile = codecs.open(os.path.join(FLAGS.raw_dir,f),"r")
            output_tag = codecs.open(os.path.join(FLAGS.sents_dir,pmid+".tags"),"w")
            output_sent = codecs.open(os.path.join(FLAGS.sents_dir,pmid+".sents"),"w")
            for section in infile:
                tag, sents = preprocess_ab(section.rstrip())
                for s in sents:
                    output_sent.write(s+"\n")
                    output_tag.write(tag+"\n")
        except:
            print ("exception:",f)
            exception.write(f+"\n")



if __name__ == "__main__":
    main()
    print("sents and tags saved in",FLAGS.sents_dir)

