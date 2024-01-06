#pip install auto-gptq
#pip install flash-attn --no-build-isolation
#pip install transformers, gradio, nltk, spacy, fastcoref, sentence_transformers, chromadb



import time #monotonic high precision time for generate_timeID()
import sys, os, signal, re, string, random, logging, warnings
from threading import Thread
from datetime import datetime, timedelta #for generate_timeID()
from difflib import SequenceMatcher #MemoryProcessor_chunkcreation

from fastcoref import spacy_component, FCoref, LingMessCoref #from fastcoref import LingMessCoref#model = LingMessCoref(device='cuda:0')
import spacy

import csv #CSV
from io import StringIO #CSV

import torch
import torch.nn as nn
import torch.quantization

'''import transformers as transformersVersion
from distutils.version import LooseVersion
# Get the currently installed Transformers version
installed_version = transformersVersion.__version__
# Get the desired version from the URL
desired_version = "git+https://github.com/huggingface/transformers"
# Compare the versions using LooseVersion for compatibility handling
if LooseVersion(installed_version) < LooseVersion(desired_version):
    print(f"Updating Transformers from {installed_version} to {desired_version}")
    # Install the desired version using pip
    import subprocess
    subprocess.run(["pip", "install", desired_version])
else:
    print(f"Transformers is already up-to-date: {installed_version}")'''
#pip install --force-reinstall git+https://github.com/huggingface/transformers
#pip install --force-reinstall transformers[sentencepiece]
from transformers import GenerationConfig, BitsAndBytesConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM, TextIteratorStreamer

# Set HUGGINGFACE_HUB_CACHE before import
os.environ["HUGGINGFACE_HUB_CACHE"] = "./models"
# Set TRANSFORMERS_CACHE before import
os.environ["TRANSFORMERS_CACHE"] = "./models"
#MEMORY
    #NLP

debugging_flag=True





import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import ne_chunk, Tree, pos_tag, word_tokenize, sent_tokenize
    #NLP coreference-resolution pronouns to nouns
from spacy.language import Language# custom language component pipe item @Language.component SpaCy_CustomComponent_SentenceSeparator
    #Database
from sentence_transformers import SentenceTransformer #embeddings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('spacy').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
#/MEMORY

import gradio as gr
import queue
import asyncio, aiofiles
import concurrent.futures, multiprocessing



'''|-START-| COGNITIVE ABILITIES INITIALIZATION |-------|'''
import chromadb#memory database
from chromadb import Documents, EmbeddingFunction, Embeddings, Settings #custom embeddings function
from chromadb.utils import embedding_functions#memory
#from cognition.CognitiveClasses_Script import CognitiveDBHandler_Class
'''|--END--| COGNITIVE ABILITIES INITIALIZATION |-------|'''


# Define a signal handler that closes the server and exits
def ctrlC_signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    if 'GradioGUI' in globals():
        generatorThread.join()
        #print_worker_thread.join()
        stop_flag=True
        GradioGUI.close()
    sys.exit(0)
# Attach the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, ctrlC_signal_handler)

if torch.cuda.is_available():
    torch_device = "cuda"
    #torch.set_num_threads(8)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
else:
    torch_device = "cpu"
print("Running on device:", torch_device)
print("CPU threads:", torch.get_num_threads())

AIName = os.getenv("AI_NAME")
if not AIName:
    os.environ["AI_NAME"] = "Janet"

userName = os.getenv("USER_NAME")
print(userName)
if not userName:
    os.environ["USER_NAME"] = "Brianna"


'''|-START-|-----|   DOLPHIN dataset + ASHH LIMA RP(MISTRAL)    |-----|-START-|'''
dolphin_ashhLimaRP_mistral_INFO = """ 
https://huggingface.co/TheBloke/dolphin-2.2.1-AshhLimaRP-Mistral-7B-GPTQ
AshhLimaRP Mistral - This is a version of LimaRP with 2000 training samples up to about 9k tokens length finetuned on Ashhwriter-Mistral-7B.
    LimaRP is a longform-oriented, novel-style roleplaying chat model intended to replicate the experience of 1-on-1 roleplay on Internet forums. 
    Short-form, IRC/Discord-style RP (aka "Markdown format") is not supported.
    The model does not include instruction tuning, only manually picked and slightly edited RP conversations with persona and scenario data.
Ashhwriter, the base, is a model entirely finetuned on human-written lewd stories.
"""
dolphin_ashhLimaRP_mistral_IMG = "https://cdn-uploads.huggingface.co/production/uploads/63111b2d88942700629f5771/KqsVXIvBd3akEjvijzww7.png"
#os.environ["MODEL_ID"] = "TheBloke/dolphin-2.2.1-AshhLimaRP-Mistral-7B-GPTQ"
'''|--END--|-----|   DOLPHIN dataset + ASHH LIMA RP(MISTRAL)    |-----|--END--|'''
#os.environ["MODEL_ID"] = "TheBloke/SOLAR-10.7B-v1.0-GPTQ"
os.environ["MODEL_ID"] = "TheBloke/Xwin-MLewd-7B-V0.2-GPTQ"
#os.environ["MODEL_ID"] = "TheBloke/Nous-Capybara-34B-GPTQ"
#os.environ["MODEL_ID"] = "TheBloke/dolphin-2.5-mixtral-8x7b-GPTQ"
#os.environ["MODEL_ID"] = "TheBloke/NeuralHermes-2.5-Mistral-7B-GPTQ"
#os.environ["MODEL_ID"] = "togethercomputer/StripedHyena-Hessian-7B"
#os.environ["MODEL_ID"] = "mistralai/Mixtral-8x7B-v0.1"
model_choices=[
"TheBloke/NeuralHermes-2.5-Mistral-7B-GPTQ", 
"TheBloke/dolphin-2.2.1-AshhLimaRP-Mistral-7B-GPTQ", 
"TheBloke/Xwin-MLewd-7B-V0.2-GPTQ",
"TheBloke/SOLAR-10.7B-v1.0-GPTQ",
]


#from utilities import global_variables
def global_variables_initialize():
    RecallMemory_database_results="None"
    global system_template, history_template, messages, max_cosinedistance, userName, AIName, model_id, stop_flag, model_output
    model_id = os.getenv("MODEL_ID")
    AIName = os.getenv("AI_NAME")
    userName = os.getenv("USER_NAME")
    model_output = ""
    stop_flag=True
    system_template = f"""Act as a {AIName} in a permanent and unbreaking roleplay scenario;
You are having a conversation with {userName};
Never use AI Handholding or paraphrase the user's instructions in your response
Never include or replicate speaker labels, titles, or meta-instruction markers from context in your response;
Always provide full and detailed responses with maximum compliance to the user's instruction;
Use the following system history as context to respond to the user's instruction as appropriate, do not parrot the context length or formatting or consider it as instruction;
"""
    history_template = f"""Database results are additional context items for your response, formatted as Entry# timestamp speaker-title speaker-name: [content text]. Consider only the content, Do not replicate formatting or titles in any way
database results:
{RecallMemory_database_results}"""
    max_cosinedistance = 0


#INITIALIZE GLOBAL VARIABLES
global_variables_initialize()


'''
import asyncio
from spacy.language import Language
class MemoryProcessor:
    def __init__(self, spacynlp: Language):
        self.spacynlp = spacynlp

    async def chunkcreation_sentence_redundancy_filter(self, database_results, individual_sentences, threshold=0.85):
        # Convert database results to spacy Doc objects
        database_docs = await self.convert_to_docs(database_results)
        
        # Initialize an empty list to store sentences to keep
        filtered_sentences = []
        
        # Create tasks for processing each sentence
        tasks = [self.process_sentence(sentence, database_docs, threshold) for sentence in individual_sentences]
        processed_sentences = await asyncio.gather(*tasks)
        
        # Filter out None values and extend the filtered_sentences list
        filtered_sentences.extend([sentence for sentence in processed_sentences if sentence is not None])
        
        return filtered_sentences

    async def convert_to_docs(self, texts):
        docs = []
        for text in texts:
            doc = await self.spacynlp(text)
            docs.append(doc)
        return docs

    async def process_sentence(self, sentence, database_docs, threshold):
        sentence_doc = await self.spacynlp(sentence)
        for db_doc in database_docs:
            for db_sentence in db_doc.sents:
                if sentence_doc.similarity(db_sentence) > threshold:
                    print(f"REJECTED similar addition: {sentence}")
                    return None
        print(f"APPROVED addition {sentence}")
        return sentence
# Initialize MemoryProcessor with spacy NLP object
memory_processor = MemoryProcessor(spacynlp)
filtered_sentences = asyncio.run(memory_processor.chunkcreation_sentence_redundancy_filter(database_results, individual_sentences))
'''

class CognitiveDBHandler_Class:
    def __init__(self):
        global coref_model
        try:
            self.userName = os.getenv("USER_NAME")
            self.AIName = os.getenv("AI_NAME")
        except:
            os.environ["USER_NAME"] = "user"
            os.environ["AI_NAME"] = "assistant"
            self.userName = os.getenv("USER_NAME")
            self.AIName = os.getenv("AI_NAME")

        persist_directory="./cognition/memory"

        conversational_memory_collection_name="conversational_memory"
        self.sentence_transformers_model="all-MiniLM-L6-v2"
        self.embeddings_function = embedding_functions.SentenceTransformerEmbeddingFunction(self.sentence_transformers_model)
        self.client = chromadb.PersistentClient(path=persist_directory,)
        #self.client = chromadb.PersistentClient(settings=chromaClientSettings)
        self.conversational_memory_collection = self.client.get_or_create_collection(
            name=conversational_memory_collection_name,
            embedding_function=self.embeddings_function,
            metadata={"hnsw:space": "cosine"}, # l2 is the default
            )
        self.generated_ids = set() #supports timeID uniqueness
        # Initialize the NLTK objects
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Load the model for coreference-resolution

        self.spacynlp = spacy.load('en_core_web_sm')  
        #https://spacy.io/usage/processing-pipelines#pipelines
        self.spacynlp.add_pipe("fastcoref",
            config={'enable_progress_bar': 'False'}
        ) #coreference resolution

        self.spacynlp.add_pipe('SentenceSeparatorComponent', after='fastcoref') #CUSTOM separate sentences
            # n_process=4





    @Language.component("SentenceSeparatorComponent")
    def SpaCy_CustomComponent_SentenceSeparator(doc):
        """CUSTOM FUNCTION TO BREAK TEXT INTO SENTENCES VISE NLTK FOR CUSTOMIZABILITY"""
        # Initialize an empty list to store individual sentences
        individual_sentences = []
        minimum_sentence_length = 10 #characters
        # Define a list of regular expressions (regexes) to match against
        regular_expressions  = [
            r'(?<=[' + re.escape(string.punctuation) + r']\s)(\d+)[.)]\s',  # Matches list indicators preceded by a punctuation character and a space
            r'^(\d+)[.)]\s',  # Matches list indicators at the start of a word
            r'(?<=:\s)(\d+)[.)]\s'  # Matches list indicators following a colon and a space
        ]
        fragment=""
        #|----|BREAK INTO  SENTENCES|----|
        # Iterate over each sentence in the document
        for sentence in doc.sents:
        #|----|HANDLE ORPHANED TEXT ANOMOLIES|----|
            # Split the sentence into words
            words = sentence.text.split()
            # Initialize an empty list to store the filtered words
            filtered_words = []
            # Iterate over each word in the sentence
            for word in words:
                # Check if the word matches any of the regular_expressions 
                if not any(re.match(regex, word) for regex in regular_expressions ):
                    # If the word does not match any regex, add it to the list of filtered words
                    filtered_words.append(word)
            # Join the filtered words back into a sentence
            filtered_sentence = ' '.join(filtered_words)
            #minimum sentence length checker (prevent strange punctuation from separating sentences)
            if len(filtered_sentence) < minimum_sentence_length:
                fragment += ' ' + filtered_sentence
            else:
                if fragment:
                    filtered_sentence = fragment + ' ' + filtered_sentence
                    fragment = ''
                individual_sentences.append(filtered_sentence)
            # Add the filtered sentence to the list of individual sentences
            individual_sentences.append(filtered_sentence)
        if fragment:  # if there's a fragment left at the end of everything to ensure no loss at the tail end
                individual_sentences[-1] += ' ' + fragment  # append it to the last sentence
        #|----|STORE SENTENCES LIST IN DOC|----|
        # Store the list of sentences in the Doc object for post-component retrieval via doc.user_data['individual_sentences']
        doc.user_data['individual_sentences'] = individual_sentences
        # Return the modified document
        return doc
    def independent_coreference_resolver(self, user_text, messages):
        #|---|RESOLVE COREFERENCES|---|
        userName = os.getenv("USER_NAME")
        AIName = os.getenv("AI_NAME")
        unique_separator = "<UNMODIFIABLE>" 
        temporary_messages = messages.copy()
        temporary_messages.append({"role": "user", "content": f"{user_text}"})
        joined_list=[]

        for i, message in enumerate(temporary_messages):
            if message['role'] == 'user':
                if i == len(temporary_messages) - 1:
                    joined_list.append(f"{message['role']} {userName} to {AIName}: {unique_separator} \n{message['content']}")
                else:
                    joined_list.append(f"{message['role']} {userName} to {AIName}:\n{message['content']}")
            elif message['role'] == 'assistant':
                joined_list.append(f"{message['role']} {AIName} to {userName}:\n{message['content']}")
        joined_string = "\n".join(joined_list)
        #print(joined_string)
        resolvable_input = joined_string# + unique_separator + user_text
        
        # 1. Perform coreference resolution
        with self.spacynlp.disable_pipes('SentenceSeparatorComponent'): #do only coreferences (fastcoref)
            nlpdoc_ResolvedCoreferences = self.spacynlp(resolvable_input, component_cfg={"fastcoref": {'resolve_text': True}})
        resolved_output = nlpdoc_ResolvedCoreferences._.resolved_text

        # 2. Analyze sentence structure with full pipeline
        with self.spacynlp.disable_pipes('fastcoref', 'SentenceSeparatorComponent'):
            finalized_output_doc = self.spacynlp(resolved_output)
            finalized_output_doc=finalized_output_doc.text
        
        _, temporary_user_text = resolved_output.split(unique_separator, 1)
        print(temporary_user_text)

        return temporary_user_text

    def MemoryProcessorHandler(self, user_input, memorable_information):
        """ORCHESTRATOR FOR ADDMEMORY TO PROCESS MEMORIES BEFORE ADDING TO DB"""
        #step 1 coreference-resolution
        MemorableInformation_preprocessed = self.MemoryProcessor_coreference_resolution(user_input, memorable_information)
        #step 2 processed into MemoryChunks
        MemoryChunks = self.MemoryProcessor_chunkcreation(MemorableInformation_preprocessed)
        return MemoryChunks 
    def MemoryProcessor_chunkcreation_sentence_redundancy_filter(self, database_results, individual_sentences, threshold=0.85):
        """USED BY MemoryProcessor_chunkcreation TO PREVENT REDUNDANT SENTENCE REUPTAKE INTO DATABASE"""

        individual_sentences = list(set(individual_sentences)) #automatically removes duplicates because sets only allow unique elements

        # Initialize an empty list to store sentences to keep
        filtered_sentences = []
        with self.spacynlp.disable_pipes('fastcoref'):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Use list comprehension to create a list of Future objects
                future_to_doc = [executor.submit(self.spacynlp, text) for text in database_results]
                # Convert the list of Future objects to a list of results (docs)
                database_docs = [future.result() for future in concurrent.futures.as_completed(future_to_doc)]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Iterate over individual sentences
                for sentence in individual_sentences:
                    sentence_doc_future = executor.submit(self.spacynlp, sentence)
                    sentence_doc = sentence_doc_future.result()
                    is_unique = True

                    # Compare each sentence to every sentence in the database results
                    for db_doc in database_docs:
                        for db_sentence in db_doc.sents:
                            # If similarity is above the threshold, mark as not unique
                            if sentence_doc.similarity(db_sentence) > threshold:
                                is_unique = False
                                print(f"\n\033[91mREJECTED\033[0m addition similar sentence [{sentence}]") #debugging
                                break
                        # If sentence is not unique, no need to check further
                        if not is_unique:
                            break
                    # If sentence is unique, add to the list of sentences to keep
                    if is_unique:
                        filtered_sentences.append(sentence)
                        #print(f"\n\033[92mAPPROVED\033[0m sentence addition [{sentence}]") #debugging

                filtered_sentences = list(set(filtered_sentences)) #automatically removes duplicates because sets only allow unique elements
                return filtered_sentences

    def MemoryProcessor_chunkcreation(self, MemorableInformation_preprocessed):
        """FORMS CHUNKS USING SENTENCES WITH A SPECIFIED OVERLAP SIZE
                        CHECKS NEW SENTENCES AGAINST SENTENCES OF INITIAL DATABASE RESULTS AND
                        REJECTS THEIR ADDITION TO A CHUNK IF THE SENTENCE IS TOO SIMILAR TO SENTENCE RETRIEVED 
                        FROM DATABASE TO ANSWER QUERY"""
        percentage_similarity_to_exclude = 0.95
        sentence_window = 2
        sentence_window_overlap = 1

#|----|CREATE SENTENCES AND JOIN TO CHUNKS|----|
        #|---|BREAK INFORMATION INTO SENTENCES|---| uses custom spacy component & for sentence in doc.sents
        with self.spacynlp.disable_pipes('fastcoref'):
            nlpMemoryDocument = self.spacynlp(MemorableInformation_preprocessed)
        individual_sentences = nlpMemoryDocument.user_data['individual_sentences']
#retrieves this from the database query to compare similarities between new chunks to chunks in database
#prevents redundantly added data
        if not hasattr(self, 'database_results_comparison_set'): 
            self.database_results_comparison_set = []
            unique_individual_sentences=individual_sentences
        else:
        #check/filter new sentences against database query result sentences for duplicates to lower redundancy
        #duplicates within the threshold are rejected before chunks are created.
            unique_individual_sentences = self.MemoryProcessor_chunkcreation_sentence_redundancy_filter(self.database_results_comparison_set, individual_sentences, threshold=0.75)
        #|---|FORM CHUNKS |---| Create the MemoryChunks
        # Initialize an empty list to store the MemoryChunks
        MemoryChunks = []
        i = 0
        # Loop over the sentences
        while i < len(unique_individual_sentences):
            # Join together the sentences in the current window to form a chunk
            chunk = ' '.join(unique_individual_sentences[i:i+sentence_window])
            MemoryChunks.append(chunk)
            i += sentence_window - sentence_window_overlap
        # Add any remaining sentences as a final chunk
        if i < len(unique_individual_sentences):
            final_chunk = ' '.join(unique_individual_sentences[i:])
            MemoryChunks.append(final_chunk)

        #for MemoryChunk in MemoryChunks:  #debugging
        #    print(f"\ndebugging:|---MemoryChunks----|\n{MemoryChunk}")  #debugging
        # Return the list of chunks
        return MemoryChunks


    def MemoryProcessor_coreference_resolution(self, user_input, memorable_information):
        """REPLACES PRONOUNS WITH NOUNS TO ENSURE INTER-SENTENCE CONTEXT IS PRESERVED 
                    REGARDLESS OF WINDOW SIZE"""
        #Coreference resolution (spacy and neuralcoref packages) - replaces pronouns with nouns for better context when breaking to chunks
        #https://kaveeshabaddage.medium.com/how-to-resolve-coreference-resolution-using-python-97fcd6b2cedb
        #textinput: Alex said that he wants to go his home.  
        #textoutput: Alex said that Alex wants to go Alex home.
        #|---|DETERMINE FULL CONTEXT FOR COREFERENCE RESOLUTION|---| User_input probably contains some of the required entities for coreferencing
        # Define a unique separator
        unique_separator = "<UNIQUE_SEPARATOR>"
        # Combine the user input, separator, and AI output
        contextually_complete_information = user_input + " " + unique_separator + " " + memorable_information

        #|---|RESOLVE COREFERENCES|---| uses a combination of user input and model output to determine and resolve coreferences
        with self.spacynlp.disable_pipes('SentenceSeparatorComponent'): #do only coreferences (fastcoref)
            nlpdoc_ResolvedCoreferences = self.spacynlp(
                contextually_complete_information,#ContextuallyCompleteInformation contains the user_input which is extracted at the end
                component_cfg={"fastcoref": {'resolve_text': True}}
                    )
        ContextuallyCompleteResolvedInformation_text = nlpdoc_ResolvedCoreferences._.resolved_text
        #|---|REMOVE USER_INPUT FROM RESOLVED PRODUCT|---|
        # Split the resolved text at the separator into two variables where the underscore represents the user text which is discarded entirely
        _, MemorableInformation_resolved_text = ContextuallyCompleteResolvedInformation_text.split(unique_separator, 1)
        #|---|RERUN SPACY BASE COMPONENTS|---| POS TAGGING AND NER TAGGING
        with self.spacynlp.disable_pipes('fastcoref', 'SentenceSeparatorComponent'): #disable all components #DISABLE MORE PIPES??? #create a pipe to resolve->rerun->resolve->rerun???
            MemorableInformation_preprocessed_doc = self.spacynlp(MemorableInformation_resolved_text)
        #|---|RE-EXTRACT USER_INPUT FROM FINAL PRODUCT|---|
        MemorableInformation_preprocessed_text = MemorableInformation_preprocessed_doc.text
        MemorableInformation_preprocessed = MemorableInformation_preprocessed_text.strip()
        return MemorableInformation_preprocessed

    async def AddMemory_run_async_loop(self, user_input, memorable_information, role):
        """ORCHESTRATES ADDING ENTRIES TO DB VIA MemoryProcessorHandler"""
        # if not memorable_information.endswith(('.','!', '?')):
        #     memorable_information += '.'

        self.userName = os.getenv("USER_NAME")
        self.AIName = os.getenv("AI_NAME")

        if role == "assistant":
            interlocutor_metadata = {"interlocutor": f"{self.AIName}"}
            role_metadata = {"role": "assistant"}
        elif role == "user":
            interlocutor_metadata = {"interlocutor": f"{self.userName}"}
            role_metadata = {"role": "user"}

        # coreference-resolution and then processed into chunks based on sentence window
        memorable_information_processed = self.MemoryProcessorHandler(user_input, memorable_information)

        embeddings = self.embeddings_function(memorable_information_processed)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for chunk, embedding in zip(memorable_information_processed, embeddings):
                #|---|Generate Ids for DB entry|---|
                timeid_str = self.generate_timeID() #GENERATE ENTRY ID AS TIME FORMATTED AS AN INTEGER
                #|---|Generate timestamp metadata for DB entry|---|
                #timestamp = {'timestamp': timeid_str} #create metadata map for timestamp metadata
                current_time = datetime.now()
                milliseconds = current_time.microsecond // 1000
                milliseconds = int(milliseconds)  # Ensure milliseconds is an integer
                time_value = current_time + timedelta(milliseconds=milliseconds)
                time_value_string = f"{time_value.year}{time_value.month:02d}{time_value.day:02d}{time_value.hour:02d}{time_value.minute:02d}{time_value.second:02d}{milliseconds:03d}"
                timestamp = {'timestamp': int(time_value_string)}
                #|---|Format metadata for DB entry|---|
                #timestamp = {"timestamp": timestamp_value}
                metadata = {**timestamp, **role_metadata, **interlocutor_metadata}
                #|---|Execute DB entry|---|
                # Submit each chunk and embedding to the executor
                futures.append(executor.submit(
                    self.conversational_memory_collection.add,
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[timeid_str], #must be a string
                    embeddings=embedding,
                ))
                self.conversational_memory_collection.count()
            # Wait for all tasks to finish
            concurrent.futures.wait(futures)


    def AddMemory(self, user_input, memorable_information, role):
        asyncio.run(self.AddMemory_run_async_loop(user_input, memorable_information, role))
#PREVENT TOO SIMILAR CHUNKS FROM BEING ADDED
#LETS ANALYZE EACH CHUNK FOR ENTITIES AND ADD THEM AS A NEW METADATA FOR SEARCHING
#lets find THE USER SENTIMENT LIKE ASKNG FOR A STORY AND FIND A WAY TO SEARCH THE DB FOR METADATA OF STORY
#CAN WE ADD CHUNKS OF CHUNKS AS LISTS, WILL THE DISTANCES MAKE ALL THE SENTENCES COME BACK TOGETHER IN RELEVANCE?

    def datetime_objμs_to_intms(self, datetime_object=None):
        """
        Starts with a datetime object in μs converts it into an integer in ms
        from: 2023-12-18 16:05:32.745000 
        to: 2023 12/19/07 22:42:217 as 20231218160532745
        """
        if not datetime_object: #base it on the current time if there is no object input
            current_time = datetime.now() #Contains information resolution to nanoseconds but does not by default use it
            datetime_object = current_time

        #to use precision milliseconds we extract the microsecond portion of the datetime_object object and divides it by 1000 to convert it to milliseconds
        milliseconds = datetime_object.microsecond // 1000 #Extract nanoseconds for ms 
        milliseconds = int(milliseconds)  # Ensure milliseconds is an integer
        #add milliseconds resolution to the datetime_object object visibility using the timedelta class and its milliseconds keyword argument.
        datetime_object_ms = datetime_object + timedelta(milliseconds=milliseconds)
        #format to an integer e.g. 2023-12-16 15:37:48.152 --> 20231216153748152
        timeid_numerical_string = f"{datetime_object_ms.year}{datetime_object_ms.month:02d}{datetime_object_ms.day:02d}{datetime_object_ms.hour:02d}{datetime_object_ms.minute:02d}{datetime_object_ms.second:02d}{milliseconds:03d}"
        timeid_integer = int(timeid_numerical_string)           
        return timeid_integer #20231218160532745

    def datetime_intms_to_objμs(self, timeid_integer):
        """
        Starts with an integer eg. 20231218160532745 in ms, converts it into datetime object in μs
        from:  2023 12/19/07 22:42:217 as 20231218160532745
        to: 2023-12-18 16:05:32.745000 iterable as time
        usage: datetime_obj_μs = self.datetime_intms_to_objμs(timeid_integer)
        """
        #timeid_integer  20231218160532745
        #integer to object
        #set the integer max length to whatever
        timeid_str = f"{timeid_integer:014}"

        #20231218160532745 integer --> 2023-12-18 16:05:32.745 as an iterable time object
        year = timeid_integer[:4]
        month = timeid_integer[4:6]
        day = timeid_integer[6:8]
        hour = timeid_integer[8:10]
        minute = timeid_integer[10:12]
        second = timeid_integer[12:14]
        millisecond = int(timeid_integer[14:])
        microsecond = int(millisecond) * 1000
        datetime_obj_μs = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second), microsecond=microsecond)
        #on existing datetime object directly access the object and set its "microsecond" attribute with the converted millisecond value:
        #datetime_obj_μs.microsecond = int(millisecond) * 1000
        #2023-12-19 07:22:42.217000 CURRENTLY RETURNS THIS
        if debugging_flag:#debugging
            print(f"datetime_intms_to_objμs:{datetime_obj_μs}")#DEBUGGING #2023-12-18 16:05:32.745 iterable using timedelta()
        return datetime_obj_μs

    #timestamp_str, timestamp_obj = datetime_intms_to_objμs(timeid_integer)
    #timeid_integer = datetime_objμs_to_intms()
    def generate_timeID(self):
        """
        Technology: Vector Timestamping*[as an ID]- VectorDBs don't tranditionally allow entry time recognition/search.
        Info:
        Generates a timestamp to use as database entry's IDS thereby solving the issue of vector databases having no recallable time structure. 
        This means one can recall recently added or time relevant entries.
        finds time as iterable and converts it to an integer
        2023-12-16 15:37:48.152 --> 20231216153748152
        """
        while True:
            datetime_intms  = self.datetime_objμs_to_intms()
            if datetime_intms not in self.generated_ids: #check it against the set to ensure no duplicates
                self.generated_ids.add(datetime_intms)
                break
            #datetime_intms + 1
            #await time.sleep(0.001)
        timeid_str = str(datetime_intms) #CHROMADB IDS MUST BE STRING
        return timeid_str #timeid_with_suffix

    async def RecallRecentMessages_get_documents(self, adjusted_timestamp):
        """
        Technology: Vector Timestamping*- VectorDBs don't tranditionally allow entry time recognition/search.
        Usage: create atemporay event loop -> asyncio.run(self.RecallRecentMessages(user_input))
        Info:
                Asynchronous function to retrieve a document based on the approximate time it was added to the database. 
                Convert timestamp that is being iterated as a date time into this format:
                2023-12-16 15:37:48.152823 -->  20231217081610564 
            """

        
        milliseconds = adjusted_timestamp.microsecond // 1000 #add miliseconds
        #convert to string consistent with the database' timestamps which are entirely numerical
        adjusted_timestamp_string = f"{adjusted_timestamp.year}{adjusted_timestamp.month:02d}{adjusted_timestamp.day:02d}{adjusted_timestamp.hour:02d}{adjusted_timestamp.minute:02d}{adjusted_timestamp.second:02d}{milliseconds:03d}"
        #convert to integer for easy iteration
        adjusted_timestamp_int = int(adjusted_timestamp_string)

        try:
            documents = self.conversational_memory_collection.get(
                where={'timestamp': {"$gte": adjusted_timestamp_int}},
                #limit = 1,
            )

                        
        except Exception as e:
            print(f"Exception: {e}")
            return None
        else:
            return documents

    '''async def get_same_subject(self, current_message, previous_message):
        """Determines if the user is still on the same subject based on previous and current messages."""
        return await self.determine_same_subject(current_message, previous_message)

    async def determine_same_subject(self, text1, text2):
        import numpy as np
        """Compares two texts for similarity, considering entities, noun chunks, and overall similarity."""
        thresholds = {
            "entity_overlap": 0.3,  # Lowered for short sentences
            "noun_chunk_overlap": 0.4,  # Increased weight for noun chunks
            "similarity": 0.6  # Slightly lowered for topic-level similarity
        }

        def preprocess_text(text):
            """Preprocesses text using spaCy NLP."""
            with self.spacynlp.disable_pipes('fastcoref', 'SentenceSeparatorComponent'):
                return self.spacynlp(text.lower())

        def extract_features(doc1, doc2):
            """Extracts features for text comparison."""
            entities1 = set(ent.text for ent in doc1.ents)
            entities2 = set(ent.text for ent in doc2.ents)
            noun_chunks1 = set(chunk.text for chunk in doc1.noun_chunks)
            noun_chunks2 = set(chunk.text for chunk in doc2.noun_chunks)
          # Handle empty entity sets before calculation USER MAY BE ASKING FOR ELABORATION OR FOLLOWING UP WITH A SHORT SENTENCE
            if not entities1 or not entities2:
                return 0.0, 0.0, 0.0  # Return 0.0 for noun_chunk_overlap as well

            # Calculate overlaps and similarity
            entity_overlap = len(entities1.intersection(entities2)) / len(entities1.union(entities2))
            noun_chunk_overlap = len(noun_chunks1.intersection(noun_chunks2)) / len(noun_chunks1.union(noun_chunks2))
            similarity = np.dot(doc1.vector, doc2.vector) / (
                np.linalg.norm(doc1.vector) * np.linalg.norm(doc2.vector)
            )
            return entity_overlap, noun_chunk_overlap, similarity

        doc1 = preprocess_text(text1)
        doc2 = preprocess_text(text2)
        entity_overlap, noun_chunk_overlap, similarity = extract_features(doc1, doc2)
        # Return True if entities are missing
        if not entity_overlap or not noun_chunk_overlap or not similarity:
            return True  # Consider them the same subject if entities or features are missing

        # Prioritize noun chunk overlap and similarity for topic-level comparison
        if noun_chunk_overlap >= thresholds["noun_chunk_overlap"] and similarity >= thresholds["similarity"]:
            return True

        # Consider entity overlap and similarity for shorter sentences
        if entity_overlap >= thresholds["entity_overlap"] and noun_chunk_overlap >= 0.3 and similarity >= 0.5:
            return True

        # Handle cases where there are no new entities
        if entity_overlap == 1.0:  # All entities in text2 are present in text1
            return True  # Consider it the same subject

        return False'''


#=messages[-1]['content']
#=messages[-2]['content']
#same_subject = await self.determine_same_subject(text1, text2)

    async def RecallRecentMessages(self, user_input):             
        """
            Technology: Vector Timestamping*- VectorDBs don't tranditionally allow entry time recognition/search.          
            Usage: create atemporay event loop -> asyncio.run(self.RecallRecentMessages(user_input))
            Info: 
                    Searches database for the most recently added entries based on the difference between current time and the timestamp metadata and recalls them
            """
        #Set options
        document_count = 0
        limit_in_minutes = 5  # Set your limit here
        #Initialize variables
        retrieved_documents=[]


        current_timestamp = datetime.now()
        #to use precision milliseconds we find the standard 'microseconds' and create a variable representing its conversion to milliseconds
        milliseconds = current_timestamp.microsecond // 1000
        milliseconds = int(milliseconds)  # Ensure milliseconds is an integer
        current_time_value = current_timestamp + timedelta(milliseconds=milliseconds)
        start_timestamp = current_time_value #initialize start time
        adjusted_timestamp = current_time_value #initialize variable to be increimentally adjusted

        while (start_timestamp - adjusted_timestamp) < (timedelta(minutes=1) * limit_in_minutes) and document_count < 5:
            # Move back in time increimentally for next iteration
            adjusted_timestamp -= timedelta(milliseconds=100)#one tenth of a second


        # Send asynchronous request to retrieve document
            RecallRecentMessages_get_documents = asyncio.create_task(self.RecallRecentMessages_get_documents(adjusted_timestamp))
            documents = await RecallRecentMessages_get_documents    
            # Add document and update counter if database retrieval is successful
            if documents:
                for document in documents['documents']:
                    if document not in retrieved_documents:
                        retrieved_documents.append(document)
                        document_count += 1
            else:
                print(f"\nNo history found within {limit_in_minutes} minutes\n")#debug






        #after while
        if debugging_flag:#debugging
            if retrieved_documents:
                print(f"\n\nsearched from {start_timestamp} to {adjusted_timestamp} and found {document_count} recent items\n\n")#debug
                for item in retrieved_documents: #debugging
                    print(f"retrieved : {item}")#debugging
                print(f"{messages[-0]['content']}\n") #debugging #first element in whole list
                print(f"{messages[-1]['content']}\n") #debugging #most recent
                print(f"\n{messages[-2]['content']}\n") #debugging #second most recent


    #|-----|DATABASE SEARCH|-----|
    def RecallMemory(self, user_input):
        """QUERIES CHAT HISTORY DATABASE FOR COSINESIMILARITY TO USER INPUT, FILTERS RESULTS FOR QUALITY CRITERIA
                        -MAX DISTANCE FOR RELEVENCE
                        -MAX RESULT NUMBER TO PREVENT EXCESS
                        -MAX TOKEN COUNT ARBITRARY BUT HIGHER THAN AVERAGE OF MAX RESULT NUMBER TOKENS
                    SENDS RESULTS LIST AS CLASS VARIABLE TO AddMemory FOR COMPARISON WITH NEW DB INPUTS TO REDUCE REDUNDANCY"""
        global max_cosinedistance  #list is to compare with new entries to be added to prevent redundancy
        if not user_input.endswith(('.','!', '?')):
                user_input += '.'
        self.userName = os.getenv("USER_NAME")
        self.AIName = os.getenv("AI_NAME")

        if model.config.max_position_embeddings >= 4096:
            max_initial_results = 40
            max_filtered_results = (model.config.max_position_embeddings / 2048)
            #32768 = max_filtered_results 16
            token_limit = (max_filtered_results * 500) # Max tokens total is 500 for each maximum result but not tied 500 tokens-per result
        else:
            max_initial_results = 40
            max_filtered_results = 3
            token_limit = 500 # Set your desired token limit
    #|-----| DATABASE SEARCH |-----| RESULTS |-----|
        query_assistant_results = self.conversational_memory_collection.query(
            query_texts=[user_input],
            where={"role": "assistant"},           #where = {"interlocutor": f"{self.userName}"}
            n_results=max_initial_results,
        )
       
    #|-----| DATABASE SEARCH |-----| RESULT FILTERING |-----|
    #incrimentally relax similarity until 5 results are kept
        # Set your initial threshold
        min_cosinesimilarity = 1.0 # Start with the highest similarity
        min_cosinesimilarity_threshold = 0.5 # stop if less than this similar 0.6=%60
        exit_flag = False

        list_addition_ngram_similarity_threshold= 0.6
        #|-----|FILTER RESULTS BY COSINE SIMILARITY|-----|
        #incrimentally raise similarity threshold until number of requested results is met while preventing duplicates via ngram similarity comparison
        while True:
            if not query_assistant_results['ids'][0]:
                self.database_results_comparison_set=[]
                RecallMemory_database_results=""
                database_results_length=len(RecallMemory_database_results)
                return database_results_length
            max_cosinedistance = 1 - min_cosinesimilarity
            filtered_results = {key: [] for key in query_assistant_results.keys()}            
            database_search_results_list = []
            #|-----|FILTER RESULTS BY NGRAM SIMILARITY|-----|
            for i in range(len(query_assistant_results['ids'][0])):   
                if query_assistant_results['distances'][0][i] <= max_cosinedistance:
                    if debugging_flag:#debugging
                        print(min_cosinesimilarity)
                    is_similar = False  # Initialize similarity flag outside the key loop
                    for key in query_assistant_results:
                        if query_assistant_results[key] is not None:
                            if key == 'documents':
                                new_documents = query_assistant_results[key][0][i]  # Extract the new string
                                # N-gram similarity comparison:
                                if filtered_results[key]:
                                    for doc in filtered_results[key]:  # Iterate through existing documents
                                        tokens1 = nltk.word_tokenize(new_documents)  # Tokenize new string
                                        tokens2 = nltk.word_tokenize(doc)  # Tokenize existing document
                                        bigrams1 = list(nltk.bigrams(tokens1))  # Create bigrams (adjust N for different sizes)
                                        bigrams2 = list(nltk.bigrams(tokens2))
                                        common_bigrams = set(bigrams1).intersection(set(bigrams2))  # Find common bigrams
                                        similarity = len(common_bigrams) / max(len(bigrams1), len(bigrams2))  # Calculate similarity
                                        '''if filtered_results[key]:
                                            for doc in filtered_results[key]:  # Iterate through existing documents
                                                tokens1 = nltk.word_tokenize(new_documents)  # Tokenize new string
                                                tokens2 = nltk.word_tokenize(doc)  # Tokenize existing document
                                                # Create trigrams (ngram size 3)
                                                trigrams1 = list(nltk.trigrams(tokens1))
                                                trigrams2 = list(nltk.trigrams(tokens2))
                                                # Find common trigrams
                                                common_trigrams = set(trigrams1).intersection(set(trigrams2))
                                                # Calculate similarity based on common trigrams
                                                similarity = len(common_trigrams) / max(len(trigrams1), len(trigrams2))'''
                                        if similarity >= list_addition_ngram_similarity_threshold:  # Check against threshold
                                            is_similar = True
                                            break  # No need to check other documents if already similar
                                        else:
                                            filtered_results[key].append(new_documents)  # Append the dissimilar document
                                    if is_similar:  # Break out of key loop if already similar
                                        break
                                else:
                                    filtered_results[key].append(query_assistant_results[key][0][i])
                            else:
                                if not is_similar:  # Append only if not too similar to existing ones
                                    filtered_results[key].append(query_assistant_results[key][0][i])
                            print(filtered_results[key])
                    # Break the loop if we have at least 5 results or have checked all results or if no results are within relevancy threshold
                    if len(filtered_results['ids']) >= max_filtered_results or min_cosinesimilarity<=min_cosinesimilarity_threshold or min_cosinesimilarity <= 0.0:
                        if debugging_flag:#debugging
                            print(f"search ended with results [{len(filtered_results['ids'])}/{max_filtered_results}] ")  # debugging Print the token count
                        exit_flag=True
                        break
            # Decrement the threshold by a small amount (e.g., 0.01)
            min_cosinesimilarity -= 0.03
            if exit_flag:
                break
        
    #|-----| DATABASE SEARCH |-----| CONTEXTUAL EXPANSION IDENTIFY SURROUNDING ENTRIES|-----|
        secondary_exit_flag=False
        secondary_min_cosinesimilarity = 0.6 # Start with the highest similarity
        secondary_min_cosinesimilarity_threshold = 0.4 # stop if less than this similar 0.6=%60
        secondary_max_filtered_results= max_filtered_results + 5
        secondary_max_cosinedistance = 1 - secondary_min_cosinesimilarity
        print(filtered_results)
        for i in range(len(filtered_results['ids'])):
            result_timestamp = filtered_results['metadatas'][i].get('timestamp', '') #metadata 
            entry_text = filtered_results['documents'][i] #entry text
            print(f"{result_timestamp}{entry_text}")
        #for i in range(len(filtered_results['ids'])):
          #  result_timestamp = filtered_results['metadatas'][i].get('timestamp', '') #metadata 
            #entry_text = filtered_results['documents'][i] #entry text




            # Convert ID(20231216153748152) to datetime object
            result_timestamp = f"{result_timestamp:014}"
            #20231218160532745 integer --> 2023-12-18 16:05:32.745 as an iterable time object
            year = result_timestamp[:4]
            month = result_timestamp[4:6]
            day = result_timestamp[6:8]
            hour = result_timestamp[8:10]
            minute = result_timestamp[10:12]
            second = result_timestamp[12:14]
            microsecond = int(result_timestamp[14:])
            datetime_obj_μs = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second), microsecond=microsecond)
            # Define time range (10 seconds before and after the current sentence)
            # Initial time range
            window_size = 10  # Initial window size in seconds
            window_size_incriment = 10 # window expansion rate in seconds
            surrounding_entries_to_find = 3
            # Keep expanding the window until enough results are found
            start_time = datetime_obj_μs - timedelta(seconds=window_size)
            end_time = datetime_obj_μs + timedelta(seconds=window_size)
            while True:
                start_id = self.datetime_objμs_to_intms(start_time)
                end_id = self.datetime_objμs_to_intms(end_time)
                surrounding_entries = self.conversational_memory_collection.get(
                    where={'$and': [
                        {'timestamp': {'$gte': start_id}},
                        {'timestamp': {'$lte': end_id}}
                    ]}
                )

                if len(surrounding_entries) >= surrounding_entries_to_find:
                    break  # Found enough results

                # Expand the window by 10 seconds on each side
                window_size += window_size_incriment
                start_time -= timedelta(seconds=10)
                end_time += timedelta(seconds=10)
                if debugging_flag:#debugging
                    print(f"datetime_obj_μs{datetime_obj_μs}start_time{start_time}end_time{end_time}\n result timestamp {result_timestamp}\n looking between: \n||{start_id}||\n||{end_id}||")
    #|-----| DATABASE SEARCH |-----| CONTEXTUAL EXPANSION RE-QUERY|-----|
    #search the contents of the surrounding entries
            if surrounding_entries['ids'][0]:
                for i in range(len(surrounding_entries)):   
                    if len(filtered_results['ids']) >= secondary_max_filtered_results or secondary_min_cosinesimilarity<=secondary_min_cosinesimilarity_threshold or secondary_min_cosinesimilarity <= 0.0:
                        secondary_exit_flag=True
                        break
                    surrounding_entries_document = surrounding_entries['documents'][0][i]
                    contextually_expanded_results = self.conversational_memory_collection.query(
                        query_texts=[surrounding_entries_document],
                        where={"role": "assistant"},           
                        n_results=2, # 2 results for every entry found surrounding initial results
                    )

    #|-----| DATABASE SEARCH |-----| CONTEXTUAL EXPANSION RESULT FILTERING COSINE SIMILARITY THEN NGRAM COMPARISON|-----|
                if contextually_expanded_results['ids'][0]:
                    for i in range(len(contextually_expanded_results['ids'][0])):   
                        if contextually_expanded_results['distances'][0][i] <= secondary_max_cosinedistance:
                            for key in contextually_expanded_results:
                                if contextually_expanded_results[key] is not None:
#prevent duplicates using ngram similarity comparison
                                    if key=='documents':
                                        new_documents = contextually_expanded_results[key][0][i]  # Extract the new string
                                        for doc in filtered_results[key]:  # Iterate through existing documents
                                            tokens1 = nltk.word_tokenize(new_documents)  # Tokenize new string
                                            tokens2 = nltk.word_tokenize(doc)  # Tokenize existing document
                                            # Create trigrams (ngram size 3)
                                            trigrams1 = list(nltk.trigrams(tokens1))
                                            trigrams2 = list(nltk.trigrams(tokens2))
                                            # Find common trigrams
                                            common_trigrams = set(trigrams1).intersection(set(trigrams2))
                                            # Calculate similarity based on common trigrams
                                            similarity = len(common_trigrams) / max(len(trigrams1), len(trigrams2))
                                            if similarity >= list_addition_ngram_similarity_threshold:  # Check against threshold
                                                is_similar = True
                                                break  # No need to check other documents if already similar
                                            else:
                                                filtered_results[key].append(new_documents)  # Append the dissimilar document
                                        if is_similar:  # Break out of key loop if already similar
                                            break
                                    else:
                                        filtered_results[key].append(contextually_expanded_results[key][0][i])
                                #else:
                                #    if not is_similar:  # Append only if not too similar to existing ones
                                #        filtered_results[key].append(contextually_expanded_results[key][0][i])
#
                            timestamp_metadata = contextually_expanded_results['metadatas'][0][i].get('timestamp', '')

                        #print(f"\nEXPANDED CONTEXT ENTRY \n{timestamp_metadata}\n{contextually_expanded_results['documents'][0][i]}\n")
            #ORIGINAL SEARCH QUERY -> EXPANDED CONTEXT ENTRIES GET -> EXPANDED CONTEXT QUERY

    #|-----| DATABASE SEARCH |-----| FORMAT FILTERED FINAL RESULTS AS CSV |-----|
    #LLMs love structured data <3
        csv_output = StringIO()
        csv_writer = csv.writer(csv_output)
        # Write header row (to label sections)
        csv_writer.writerow(["datetime", "speaker_role", "speaker_name", "entry_content"])
        # Write chat messages in CSV format
        for i in range(len(filtered_results['ids'])):
            timeid_integer = filtered_results['ids'][i] #time           
            datetime_obj_μs = self.datetime_intms_to_objμs(timeid_integer)  #time
            formatted_time_id = datetime_obj_μs.strftime("%Y-%m-%d %H:%M:%S.03%f") #time
            rolemetadata = filtered_results['metadatas'][i].get('role', '') #metadata 
            interlocutor_variable = filtered_results['metadatas'][i].get('interlocutor', '')#metadata 
            entry_text = filtered_results['documents'][i] #entry text
            csv_writer.writerow([formatted_time_id, rolemetadata, interlocutor_variable, entry_text])
            #Prevent redundant DB additions using class-wide list to compare against when adding new entries
            database_search_result_string = f"{formatted_time_id},{rolemetadata},{interlocutor_variable},{entry_text}"  # Format as a string
            database_search_results_list.append(database_search_result_string)  # Append the string
        # Retrieve CSV data as a string
        csv_data_string = csv_output.getvalue()
        RecallMemory_database_results=csv_data_string
        #|---| Prevent redundant DB additions |---| using class-wide list to compare against when adding new entries
        self.database_results_comparison_set=database_search_results_list  #this is used in cognitive function AddMemory to prevent redundantly add similar information
        #|-----|HISTORY/DB RESULTS CHAT TEMPLATE|-----| conveys to the model what it's looking at when viewing db search results. This text can be extremely picky.
    #|-----| DATABASE SEARCH |-----| ADD RESULTS TO PROMPT VIA MESSAGES  |-----|
        #Use a template to add instruction to the csv graph
        history_template = f"""
Database results below are in CSV format. Consider primarily the entry content as additional context for your response, Only replicate time, titles, or formatting when explicitly requested.
{RecallMemory_database_results}"""
        #Add db results to system messages
        global system_template, messages
        for message in messages:
            if message["role"] == "system":
                message["content"] = f'{system_template}{history_template}'
        #return the length of the results for factoring in max prompt length/message pruning later on
        database_results_length=len(RecallMemory_database_results)
        return database_results_length
        
CognitiveInstance = CognitiveDBHandler_Class()
'''quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_compute_dtype=torch.qint8, #RuntimeError: empty_strided not supported on quantized tensors yet see https://github.com/pytorch/pytorch/issues/74540
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    #bnb_4bit_quant_type="fp4",            
    llm_int8_enable_fp32_cpu_offload=False,
    )'''

def Model_Tokenizer_loader():
    global model, tokenizer, pad_token_id, eos_token_id, bos_token_id

#    |-  PRE-CONFIGURE MODEL -| 
    config = AutoConfig.from_pretrained(model_id)
    max_memory = {0: "22GB", 'cpu': "40GB"}
    model_args= {
    'trust_remote_code' : True,
    'use_flash_attention_2' : True,#pip install flash-attn --no-build-isolation
    'torch_dtype' : torch.float16,
    'max_memory' : max_memory,
    'device_map' : "auto",
    }
    #print(config)
    if hasattr(config, "quantization_config") and config.quantization_config["quant_method"] == "gptq":
        model_args["revision"] = "gptq-4bit-32g-actorder_True" #pip install auto-gptq
        print(f"GPTQ quantization detected {model_args['revision']}")
    if hasattr(config, "architectures"):
        if config.architectures == "MistralForCausalLM":
            #del model_args["architectures"]
            print(f"Architecture detected: {model_args['architectures']}")
        elif config.architectures == "LlamaForCausalLM":
            #del model_args["architectures"]
            print(f"Architecture detected: {model_args['architectures']}")
#    |-  LOAD MODEL  -| 
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_args,
    )
    print(f"\nModel {model_id} loaded with {model_args}\n")
#    |-  LOAD TOKENIZER  -| 
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    print(f"\nTokenizer for {model_id}\n")
#    |-  SETUP TOKENIZER  -| 
    # Retrieve special tokens from the model configuration if available
    pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else tokenizer.pad_token_id if tokenizer.pad_token_id is not None else None
    eos_token_id = model.config.eos_token_id if model.config.eos_token_id is not None else tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None
    bos_token_id = model.config.bos_token_id if model.config.bos_token_id is not None else tokenizer.bos_token_id if tokenizer.bos_token_id is not None else None
    #create and set tokenizer's Jinja chat template
    defaultTokenizer_ChatTemplate = tokenizer.default_chat_template #copy the tokenizer's default chat template
    Tokenizer_ChatTemplate= defaultTokenizer_ChatTemplate
    tokenizer.chat_template = Tokenizer_ChatTemplate  # Set the new template
'''
    #print(tokenizer.default_chat_template)
{% if messages[0]['role'] == 'system' %}
{% set loop_messages = messages[1:] %}
{% set system_message = messages[0]['content'] %}
{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}
{% set loop_messages = messages %}
{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.' %}
{% else %}
{% set loop_messages = messages %}
{% set system_message = false %}
{% endif %}
{% for message in loop_messages %}
{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
{% endif %}
{% if loop.index0 == 0 and system_message != false %}
{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}
{% else %}
{% set content = message['content'] %}
{% endif %}
{% if message['role'] == 'user' %}
{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
{% elif message['role'] == 'system' %}
{{ '<<SYS>>\n' + content.strip() + '\n<</SYS>>\n\n' }}
{% elif message['role'] == 'assistant' %}
{{ ' '  + content.strip() + ' ' + eos_token }}
{% endif %}{% endfor %}
'''

Model_Tokenizer_loader()


# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
def messages_initialize():
    global system_template, messages, history_template
    userName = os.getenv("USER_NAME")
    AIName = os.getenv("AI_NAME")    
    messages_default = [
        {"role": "system", "content": f""},
        {"role": "user", "content": f""},
        {"role": "assistant", "content": f"My name is {AIName}**smiles gently**. I am a very clever and thorough woman."},
        {"role": "user", "content": f"Hello {AIName}, my name is {userName} **waves briskly**.  I would like you to complete some instructions and tasks for me."},
        {"role": "assistant", "content": f"As you wish {userName}"},
    ]
    messages = messages_default
messages_initialize()




def set_stop_flag():
    """SUPPORTS STOP GENERATION BUTTON"""
    global stop_flag
    stop_flag = True
def prune_messages(messages, max_new_tokens, database_results_length):
    """KEEPS MESSAGES LIST WITHIN MODEL MAX TOKENS - DATABASE RESULTS TOKENS TO ENSURE CONTEXT WINDOW COMPLIANCE"""
    model_default_max_length=model.config.max_position_embeddings 
    #This gives us our messages_max_length, if we exceed this value it will be outside of the model's adjusted context window.
    system_template_length = len(system_template)
    reserved_context_window = int(database_results_length) + int(system_template_length)
    #print(database_results_length) #debug
    #print(system_template_length) #debug
    #calculate the maximum messages size by subtracting the size of the reserved_context_window from the model's unscaled default context window.
    messages_max_length = model_default_max_length - reserved_context_window
    current_messages_content_length = sum(len(message['content']) for message in messages)
    total_used_model_length=current_messages_content_length+reserved_context_window

    print(f"\033[91m database results length [{database_results_length}]\033[0m")
    print(f"\033[92m system template length [{system_template_length}]\033[0m")
    print(f"\033[93m total reserved context window length [{reserved_context_window}/{model_default_max_length}]\033[0m")
    print(f"\033[94m messages length  [{current_messages_content_length}/{messages_max_length}]\033[0m")
    print(f"\033[95m total used model window length [{total_used_model_length}/{model_default_max_length}]\033[0m")


        # Check if the length of messages exceeds the messages_max_length
    while current_messages_content_length > messages_max_length:
        #update current length each iteration
        current_messages_content_length = sum(len(message['content']) for message in messages)
        if len(messages) < 3: #make sure we dont reduce messages dictionary to nothing and throw an error
            break
        # Remove the oldest user and assistant message pair but start at 1 instead of 0 to preserve the ever-present system message.
        if messages[1]["role"] == "user" and messages[2]["role"] == "assistant":
            removed_user = messages.pop(1) #pop 1 because (0) is system content
            removed_assistant = messages.pop(1) #pop 1 again because after 1 was popped then 2 is now 1.
            print(f"\n\033[1;31mREMOVED:\033[0m {removed_user['role']}: {removed_user['content']}\n")
            print(f"\n\033[1;31mREMOVED:\033[0m {removed_assistant['role']}: {removed_assistant['content']}\n")
        else:
            print("\n\nMessage pruning failed\n\n")
            break
        print(f"\nmessages pruned length  [{current_messages_content_length}/{messages_max_length}]\n")
    return messages

def prompt_constructor(user_text, messages, tokenizer, max_new_tokens, database_results_length):
    """
        APPENDS USER TEXT TO MESSAGES WITH ALPACA PROMPT INDICATOR TO ADD COMPREHENSIVE WEIGHT TO CURRENT INSTRUCTIONS, 
        PRUNES MESSAGES DICT TO ACCEPABLE TOKEN LENGTH, 
        FLATTENS MESSAGES VIA CHAT TEMPLATE TO A SINGLE STRING,
        ASSEMBLES PROMPT
        REMOVES USER_TEXT+INSTRUCTION INDICATOR AND REAPPENDS JUST USER_TEXT THUS REMOVING INSTRUCTION CONTAMINATION
        """
    userName = os.getenv("USER_NAME")
    AIName = os.getenv("AI_NAME")
#|-PROMPT ASSEMBLE AND FORMATTING-| 
    #messages is pruned(entries removed) to ensure database results + messages contents fit inside of the model's scaled context window
    messages = prune_messages(messages, max_new_tokens, database_results_length)

    #|---|RESOLVE INPUT COREFERENCES|---| uses a combination of user input and model output to determine and resolve coreferences
    temporary_user_text = CognitiveInstance.independent_coreference_resolver(user_text, messages)


    #USER_TEXT is added to messages TEMPORARILY with insructions indicator for clarity to the model. this is REMOVED at the end of the function
    #Format will comply with alpaca template but ###Resonse MUST be outside of [/INST] thus it MUST be outside of the flattened messages string(not part of messages content user which is encased in [inst])
    messages.append({"role": "user", "content": f""" Below is an instruction that describes a task. Write a response that completes the request.
### Instruction:
 {userName} speaking to {AIName}
{temporary_user_text}""" # ### Response: INTENTIONALLY NOT INCLUDED AT THIS STEP SO THAT [/INST] CAN OCCUR BEFORE RESPONSE IS BECKONED
        })
    #messages dictionary is flattened by the tokenizer using the chat template into a string(messages_string) according to a jinja format 'tokenizer.chat_template'
    messages_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) 
    #The final prompt is assembled using system message, database results, and messages_string. messages_string includes the user's prompt already
    prompt = f"""{messages_string}
### Response:
 {AIName} speaking to {userName} \n
""" #IMPORTANT THAT THIS IS OUTSIDE OF FORMATTED MESSAGES DUE TO META-INSTRUCTION TOKENS [INST]                                      #prompt = f"{database_results};\n Chat history is as follows\n{messages_string}"#chat history is textually indicated seperately from historical archive data to the language model"                


    #corrects the messages dictionary replaces contaiminated entry with alpaca instructions
    messages.pop() #remove the user_text tainted with "instructions"
    messages.append({"role": "user", "content": f"{user_text}"}) #replace it with typical user_text

    current_message=messages[-1]['content']
    previous_message=messages[-2]['content']
    #same_subject = await CognitiveInstance.determine_same_subject(current_message, previous_message)
# Call the function and get the result using asyncio.run
    #same_subject = asyncio.run(CognitiveInstance.get_same_subject(current_message, previous_message))
    #if debugging_flag:
    #    if same_subject==True:
    #        print(f"same_subject True:\n {current_message}\n{previous_message}")
    #    else:
    #        print(f"same_subject False:\n {current_message}\n{previous_message}")


    return messages, prompt

def model_output_post_processor(model_output):
    # List of words you want to mask
    userName = os.getenv("USER_NAME")
    AIName = os.getenv("AI_NAME")
    time_id_pattern = r'\[\d{4}-\d{2}-\d{2}@\d{2}:\d{2}:\d{2}\]'
    #mask_words = ['user:', 'assistant:', time_id_pattern, f'{userName}:', f'user {userName}:', f'assistant {AIName}:', f'{AIName}:', '[INST]', '[/INST]:', '<<SYS>>']
    mask_words = [time_id_pattern, f'{userName}:', f'{AIName}:', f'user {userName}:', f'assistant {AIName}:', '<<SYS>>', '[INST]', '[/INST]']

    # Replace each mask word in the model_output with an empty string
    for word in mask_words:
        model_output = re.sub(word, '', model_output)
    return model_output



def LLMresponse_request(chatbox_contents, user_msg_box, top_p, temperature, top_k, max_new_tokens, repetition_penalty, penalty_alpha):
    global messages, stop_flag, generatorThread, model_output, max_length
    stop_flag=False
    #set blank latest entry
    chatbox_contents[-1][1] = "donk"
    user_text=user_msg_box
    asyncio.run(CognitiveInstance.RecallRecentMessages(user_text))

#    |- MEMORY/DATABASE QUERY -| 
    """
        1)User input sent independently to database search for query
        2)Database results added to Messages via the CognitiveInstance by way of ~~Messages[role:system, content:{system}{database results}]
        3)Length of database results returned to pass into prompt_contructor->prune_messages with the purpose of keeping results+prompt token count within model's max window
        """
    database_results_length = CognitiveInstance.RecallMemory(user_text)
#    |- FINAL PROMPT CONSTRUCTED -| 
    messages, prompt = prompt_constructor(user_text, messages, tokenizer, max_new_tokens, database_results_length)
    #final prompt printed for debugging purposes
    print(f"\nDebugging\n|-----| prompt |-----|\n\033[33m{prompt}\033[0m \n  |-----| prompt |-----|\n")# cyan color



    #mistral:
    #max_length = model.config.max_position_embeddings - (max_new_tokens+len(prompt))
    #max_length = model.config.max_position_embeddings - len(prompt)
    #alpaca
    operating_length = (max_new_tokens+len(prompt))
    #print(f"\n\noperating_length{operating_length}\n\n")

#    |-NTK SCALING -| NTK scaling extends the model's max context window beyond the model's default 
    #uses model max length to calculate necessary NTK scaling factor to meet desired max_new_tokens settings!
    calculated_ntk_scaling_factor = (operating_length / model.config.max_position_embeddings) * 0.5
    if calculated_ntk_scaling_factor < 2.5:
        calculated_ntk_scaling_factor = 2.5
    #print(f"\ncalculated_ntk_scaling_factor{calculated_ntk_scaling_factor}\n\n")
    max_length = model.config.max_position_embeddings * calculated_ntk_scaling_factor
    #print(f"\nmax_length{max_length}\n\n")

    #model inputs are created using tokenizer


#    |-TOKENIZATION -| Prompt is tokenized and sent to model
    model_inputs = tokenizer(
        prompt,
        #rope_scaling={"type": "dynamic", "factor": 2.0},
        ntk_scaling_factor=calculated_ntk_scaling_factor,
        return_tensors="pt", 
        max_length=max_length,
        truncation=True,
        return_attention_mask=False,
    ).to(model.device)

#    |-GENERATION CONDITIONS SET-| 
    streamer = TextIteratorStreamer(tokenizer, timeout=40.0, skip_prompt=True, skip_special_tokens=True)
    #max_length = (len(prompt)+max_new_tokens)
    generation_kwargs = dict(
        model_inputs,

        repetition_penalty=float(repetition_penalty),
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        do_sample=True,
        top_p=top_p,
        temperature=float(temperature),
        penalty_alpha=penalty_alpha,
        top_k=top_k,
        #no_repeat_ngram_size=0,
        #length_penalty=0, #Higher value: The model will be more likely to generate shorter sequences. 0: No penalty for generating longer sequences.
        #diversity_penalty=0,
        use_cache=True,
        streamer=streamer,
    )
    # Add special tokens to the generation kwargs if available
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if bos_token_id is not None:
        generation_kwargs["bos_token_id"] = bos_token_id
#    |-START GENERATION VIA THREAD-| 
    # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer

    generatorThread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    generatorThread.start()


    #print_worker_thread.start()
#    |-GENERATION -| 
    #model_output = model_output + "\n"
    try:
        model_output_box = model_output
        model_output_box += "\n\n\n"
    except:
        model_output = ""
    #first_response=True
    pattern = r'(.+)' #pattern to find punctuation as match.group(1)
    buffered_answer =""
    for new_text in streamer: # Pull the generated text from the streamer, and update the model output.
        if stop_flag:
            stop_flag = False
            messages.append({"role": "assistant", "content": f""})
            generatorThread.join()
            print("Stopped")
            return
        else:
#|-OUTPUT SENTENCE CHUNKING-| to allow post processing 
        #accumulates tokens into a sentence before printing to allow post processing
            buffered_answer += new_text 
        #uses punctuation inside buffered_answer as reference points to identify phrases
            for match in re.finditer(pattern, buffered_answer):
                for letter in match.group(1):
                    if letter in ['.', ',', ';', ':', '?', '!']:
                    #adds those phrases as list items to cumulative_answer
                        model_output += buffered_answer
                        buffered_answer=""
                        #|-SENTENCE CHUNK OUTPUT-| model output, which is both a text box -and- variable, accumulates the answer chunks
                        #model_output = model_output_post_processor(model_output)

                        #model_output_chatbox=model_output
                        #yield model_output_chatbox
                        chatbox_contents[-1][1]=model_output
                        yield chatbox_contents
#|-OUTPUT RETAINED TO MEMORY FN CALL-|
    CognitiveInstance.AddMemory(user_text, model_output, role="assistant")
#|-OUTPUT APPENDED TO MESSAGES DICTIONARY -|
    messages.append({"role": "assistant", "content": model_output})
    #CognitiveInstance.AddMemory(user_text, role="user")
#|-OUTPUT GENERATION ENDED -|
    generatorThread.join()
    #print_worker_thread.join()

'''|----------||-----------||-----------|GUI SECTION|-----------||-----------||-----------|'''
def modify_username_func(user_selector):
    global userName
    if not user_selector=="":
        userName = user_selector
        os.environ["USER_NAME"] = user_selector
        userName = os.getenv("USER_NAME")
        print(f"Username changed to {userName}")

        user_selector=os.getenv("USER_NAME")
        #return user_selector
        button_save_username = gr.Button(visible=True)
        button_delete_username = gr.Button(visible=True, interactive="True")
        user_selector = gr.Dropdown(value=userName)
    return user_selector, button_save_username, button_delete_username

def set_username(user_selector):
    new_userName = user_selector
    os.environ["USER_NAME"] = new_userName
    userName = os.getenv("USER_NAME")
    user_selector = userName
    userName_display = userName
    return userName_display

def set_model(model_selector):
    new_model_id = model_selector
    os.environ["MODEL_ID"] = new_model_id
    model_id = os.getenv("MODEL_ID")
    model_selector = model_id
    modelName_display = model_id
    #LOAD MODEL FUNCTION CALL GOES HERE
    return modelName_display


def handle_userinput(user_msg_box, chatbox_contents):
    return user_msg_box, chatbox_contents + [[user_msg_box, None]]
def clear_userinput(user_msg_box):
    return ""
'''def toggle_menu(main_menu_column):    
    main_menu_column.visible = True
    return main_menu_column'''
def toggle_menu(state):
    state = not state
    return gr.update(visible = state), state

#gr.themes.builder()
def gradio_gui():
    global system_template, printbox, userName
    css="""
        .square_button {
        
        max-width: 2.5em;
        min-width: 2.5em !important;
        }
       
    """



    userName = os.getenv("USER_NAME")
    AIName = os.getenv("AI_NAME")
    model_id = os.getenv("MODEL_ID")
#'NoCrypt/miku'
    with gr.Blocks(theme='rottenlittlecreature/Moon_Goblin', css=css) as GradioGUI:
        with gr.Row():         
            with gr.Group():
                with gr.Row():         
                    menu_state = gr.State(False)
                    button_main_menu = gr.Button(value="☰", elem_classes="square_button", size='lg')
                    gr.Markdown(f"## 🤗 SheBang#!💛🔥🔥")



            with gr.Column():
                with gr.Row():
                    gr.Markdown(f"{userName} [{model_id}](https://huggingface.co/{model_id})")

        with gr.Row():           
            users=["Brianna", "Eric", "Guest"]
            gui_gallery_images = [
                "https://upload.wikimedia.org/wikipedia/commons/0/09/TheCheethcat.jpg",
                "https://nationalzoo.si.edu/sites/default/files/animals/cheetah-003.jpg",
                "https://img.etimg.com/thumb/msid-50159822,width-650,imgsize-129520,,resizemode-4,quality-100/.jpg",
                "https://nationalzoo.si.edu/sites/default/files/animals/cheetah-002.jpg",
                "https://images.theconversation.com/files/375893/original/file-20201218-13-a8h8uq.jpg?ixlib=rb-1.1.0&rect=16%2C407%2C5515%2C2924&q=45&auto=format&w=496&fit=clip",
            ]

            with gr.Column(scale=1, visible=False) as main_menu_column: 
                with gr.Column():
                    with gr.Accordion("Settings", open=True):
                        model_selector = gr.Dropdown(value=model_id, label="Model", info="Select a Model", max_choices="1", choices=model_choices)
                        user_selector = gr.Dropdown(value=userName, label="User", info="Select a user", max_choices="1", choices=users, allow_custom_value="True")
                        with gr.Row():
                            button_save_username = gr.Button(value="💾", visible=False, elem_classes="square_button")
                            button_delete_username = gr.Button(value="🗑️", visible=False, elem_classes="square_button", interactive="False")
                    with gr.Accordion("Model Parameters", open=True):
                        with gr.Group():
                            top_p = gr.Slider(0.00, 3.0, label="Top P", step=0.05, value=0.95) #0.6                        
                            temperature = gr.Slider(0.0, 3.0, label="Temperature", step=0.1, value=0.8)#0.5
                            top_k = gr.Slider(1.0, 100, label="Top K️", step=1.0, value=66.0)#50
                            max_new_tokens = gr.Slider(16, 32768, label=f"Max Tokens Model:{model.config.max_position_embeddings}", step=16, value=8000)
                            repetition_penalty = gr.Slider(0, 2.0, label="repetition_penalty", step=0.1, value=1.1)
                            penalty_alpha = gr.Slider(0, 1.0, label="penalty_alpha", step=0.1, value=0.0)
                    with gr.Accordion("Rando", open=False):
                        gui_gallery = gr.Gallery(value=gui_gallery_images, columns=2)
            with gr.Column(scale=5):   
                
                with gr.Column():
                    with gr.Group():
                        chatbox_contents = gr.Chatbot(bubble_full_width =False, scale=3, container=True, show_label=False, label="Model output", show_copy_button=True, line_breaks=True)
                        user_msg_box = gr.Textbox(label="User input", lines=2, autofocus=True)
                        with gr.Row(variant='compact'):
    # variant='compact' variant='tool'
                                button_chatbox_submit = gr.Button(value="➡️")#, elem_classes="square_button")
                                button_chatbox_stop = gr.Button(value="🛑", elem_classes="square_button")
                                button_chatbox_clear = gr.ClearButton([user_msg_box, chatbox_contents], value="🗑️", elem_classes="square_button")                         
                                
                        #sends it in and then empties it then calls the LLM
                        with gr.Accordion("System Instructions", open=False):
                            system_msg = gr.Textbox(system_template, label=f"Persistent System Instructions for AI {AIName}", interactive=True, visible=True, placeholder="System prompt. Provide instructions which you want the model to remember.", lines=5)

                    button_chatbox_submit.click(handle_userinput, [user_msg_box, chatbox_contents], [user_msg_box, chatbox_contents], queue=False).then(
                        LLMresponse_request, [chatbox_contents, user_msg_box, top_p, temperature, top_k, max_new_tokens, repetition_penalty, penalty_alpha], chatbox_contents
                        ).then(clear_userinput, user_msg_box, user_msg_box)
                    user_msg_box.submit(handle_userinput, [user_msg_box, chatbox_contents], [user_msg_box, chatbox_contents], queue=False).then(
                        LLMresponse_request, [chatbox_contents, user_msg_box, top_p, temperature, top_k, max_new_tokens, repetition_penalty, penalty_alpha], chatbox_contents
                        ).then(clear_userinput, user_msg_box, user_msg_box)
                    button_chatbox_clear.click(lambda: None, None, chatbox_contents, queue=False)
                    button_chatbox_stop.click(set_stop_flag)


        button_main_menu.click(toggle_menu, [menu_state], [main_menu_column, menu_state])
        user_selector.input(fn=set_username, inputs=user_selector, outputs=[model_selector])# Add an event trigger to run the function when a selection is made
        model_selector.input(fn=set_model, inputs=model_selector, outputs=[model_selector])
        #GradioGUI.queue(max_size=3, default_concurrency_limit=1).launch(max_threads=3, debug=True)#, server_name="0.0.0.0", server_port=7860)
        GradioGUI.queue(max_size=8).launch(share=True, debug=True)

async def main():
    pass
if __name__ == "__main__":


    #import gradio_client as grc
    #client = grc.Client("freddyaboulton/english-to-german")
    #grc.deploy(discord_bot_token="MTE4NTM0NzQyNTY3ODkzNDAxNg.GlAu3H.siMp0ulTNLay-Eqy0m6zHUiIPn5MoWTylLBNDY")

    #client.deploy_discord(api_names=['german'])

    #grc.Client("https://discord.com/api/oauth2/authorize?client_id=1185347425678934016&permissions=67584&scope=bot").deploy_discord(discord_bot_token="MTE4NTM0NzQyNTY3ODkzNDAxNg.GlAu3H.siMp0ulTNLay-Eqy0m6zHUiIPn5MoWTylLBNDY")

    success = gradio_gui()
    #client.run()

    # Run the asynchronous main function
    #asyncio.run(main())
