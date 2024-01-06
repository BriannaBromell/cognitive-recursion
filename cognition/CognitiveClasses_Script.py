
class CognitiveDBHandler_Class:
    def __init__(self):
        try:
            self.userName = os.getenv("USER_NAME")
            self.AIName = os.getenv("AI_NAME")
        except:
            os.environ["USER_NAME"] = "user"
            os.environ["AI_NAME"] = "assistant"
            self.userName = os.getenv("USER_NAME")
            self.AIName = os.getenv("AI_NAME")

        persist_directory="./memory"

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
        self.spacynlp.add_pipe("fastcoref") #coreference resolution
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
                        print(f"\n\033[92mAPPROVED\033[0m sentence addition [{sentence}]") #debugging

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
        unique_separator = "[UNIQUE_SEPARATOR_BUOY]"
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
        metadata = {**role_metadata, **interlocutor_metadata}

        # coreference-resolution and then processed into chunks based on sentence window
        memorable_information_processed = self.MemoryProcessorHandler(user_input, memorable_information)

        embeddings = self.embeddings_function(memorable_information_processed)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for chunk, embedding in zip(memorable_information_processed, embeddings):
                # Submit each chunk and embedding to the executor
                futures.append(executor.submit(
                    self.conversational_memory_collection.add,
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[self.generate_timeID()],
                    embeddings=embedding,
                ))
            # Wait for all tasks to finish
            concurrent.futures.wait(futures)

    def AddMemory(self, user_input, memorable_information, role):
        asyncio.run(self.AddMemory_run_async_loop(user_input, memorable_information, role))
#PREVENT TOO SIMILAR CHUNKS FROM BEING ADDED
#LETS ANALYZE EACH CHUNK FOR ENTITIES AND ADD THEM AS A NEW METADATA FOR SEARCHING
#lets find THE USER SENTIMENT LIKE ASKNG FOR A STORY AND FIND A WAY TO SEARCH THE DB FOR METADATA OF STORY
#CAN WE ADD CHUNKS OF CHUNKS AS LISTS, WILL THE DISTANCES MAKE ALL THE SENTENCES COME BACK TOGETHER IN RELEVANCE?
    def delete_entry(self, user_input):
        """ENTIRELY UNUSED"""
        deletion_query = self.conversational_memory_collection.query(
            query_texts=[user_input],
            where={"role": "assistant"},
            n_results=3
        )

        #deletion_search_results = deletion_query["documents"][0] 
        #summarize this data
        #delete results
        #add data to db
        for deletable_ids in deletion_query['ids'][0]:
            self.conversational_memory_collection.delete(
                ids=deletable_ids,
            )
            #print validation of deletion number
            number = self.conversational_memory_collection.count()
            print(f"\033[94m{number}\033[0m")  #debugging



    def MemoryCleaner(self, user_input):
        """ENTIRELY UNUSED"""
        from difflib import SequenceMatcher
        '''def group_similar_results(query_assistant_results, percentage_similarity_to_exclude=0.9):
            MemoryChunks = []
            grouped_results = []
            for result in query_assistant_results:
                chunk = result['text']  # assuming 'text' is a key in your result
                similar_group = None
                for old_chunk in MemoryChunks:
                    if SequenceMatcher(None, chunk, old_chunk).ratio() > percentage_similarity_to_exclude:
                        similar_group = old_chunk
                        break
                if similar_group is None:
                    MemoryChunks.append(chunk)
                    grouped_results.append([result])
                else:
                    index = MemoryChunks.index(similar_group)
                    grouped_results[index].append(result)
            return grouped_results
            #then we consolidate the like-entries
                """[
                    [
                        {"text": "The Aztecs and Mayans traded goods such as cacao, cotton, and feathers with The Aztecs and Mayans 's neighbors, while the Napoleonic French and Mongolians engaged in international trade, with Napoleonic French participating in the Silk Road and the Mongol Empire controlling key trade routes in Asia and Europe.."},
                    ],
                    [
                        {"text": ".Social hierarchies and political systems : Each civilization had Each civilization 's own social hierarchy and political system."},
                        {"text": "Social hierarchies and political systems : Each civilization had Each civilization 's own social hierarchy and political system.the Aztecs in Mesoamerica had a complex society with a ruling class, priests, and commoners, while the Mayans had a more egalitarian society with a focus on religion and agriculture."},
                    ],
                    [
                        {"text": "the Aztecs in Mesoamerica had a complex society with a ruling class, priests, and commoners, while the Mayans had a more egalitarian society with a focus on religion and agriculture.Napoleonic French had a feudal system with a royalty and nobility, while the Mongol Empire had a decentralized political system based on a tribal structure."},
                        {"text": "Napoleonic French had a feudal system with a royalty and nobility, while the Mongol Empire had a decentralized political system based on a tribal structure.."},
                    ],
                    [
                        {"text": ".Cultural achievements : Despite the Aztec, Mayans, Napoleonic French, and Mongolians 's differences, the Aztec, Mayans, Napoleonic French, and Mongolians contributed significantly to the development of culture and knowledge."},
                        {"text": "Cultural achievements : Despite the Aztec, Mayans, Napoleonic French, and Mongolians 's differences, the Aztec, Mayans, Napoleonic French, and Mongolians contributed significantly to the development of culture and knowledge.the Aztecs in Mesoamerica created intricate calendars and astronomical observations,"},
                    ],
                ]"""'''

        # try this instead :
        #https://www.sbert.net/examples/applications/paraphrase-mining/README.html
        # https://www.sbert.net/examples/applications/clustering/README.html
        from transformers import pipeline

        consolodate_deletion_query = self.conversational_memory_collection.query(
            query_texts=[user_input],
            where={"role": "assistant"},
            n_results=5
        )
        model="t5-small"
        #summarize this data
        summarizer = pipeline(
            task="summarization",
            model=model,
            tokenizer=tokenizer,
            )
        summarizable_string=""
        for results in consolodate_deletion_query['documents'][0]:
            summarizable_string += results
            print(results)

        print(summarizable_string)
        summary = summarizer(summarizable_string, max_length = len(summarizable_string))
        summary_text = summary[0]['summary_text']
        #add summary to db
        print(f"\033[91m{summary_text}\033[0m")
        self.AddMemory(summary_text, role="assistant")

        #delete results
        for deletable_ids in consolodate_deletion_query['ids'][0]:
            self.conversational_memory_collection.delete(
                ids=deletable_ids,
            )
            #print validation of deletion number
            number = self.conversational_memory_collection.count()

        #update instead
        for ids in consolodate_deletion_query['ids'][0]:
            collection.update(
                ids=ids,
                documents=["Kristiane Carina, a 19-year-old computer science sophomore with a 3.7 GPA"],
                metadatas=[{"source": "student info"}],
            )
        #maybe this?!
        consolodate_query = self.conversational_memory_collection.query(
            query_texts=[user_input],
            where={"role": "assistant"},
            n_results=5
        )
        for results in consolodate_deletion_query['documents'][0]:
            summarizable_string += results
            print(results)
            #find a way to summarize
            summary = summarizer(summarizable_string, max_length = len(summarizable_string))

        collection.update(
            ids=[consolodate_query['ids']],
            documents=[consolodate_query['documents']],
            metadatas=[consolodate_query['metadatas']],
        )
        #delete results
        for deletable_ids in consolodate_deletion_query['ids'][0]:
            self.conversational_memory_collection.delete(
                ids=deletable_ids,
            )
            #print validation of deletion number
            number = self.conversational_memory_collection.count()
    def get_entry_by_id(self, entry_id):
        """ENTIRELY UNUSED"""
        return self.conversational_memory_collection.get_document(id=entry_id)
    def generate_timeID(self):
        """GENRATES DB ENTRY IDS BUT USES A TIMESTAMP.MONOTONICTIME FORMAT TO ENSURE RESULTS
                        ARE NOT AMBIGUOUSLY DATED"""
        while True:
            #uses set() at init of class
            timeid = datetime.now().strftime("%Y%m%d@%H%M%S.%f")[:20]#sliced to first 20 characters of string and thus includes milliseconds
            # Monotonic high precision time component to further reduce the chance of duplication
            monotonic_suffix = str(int(time.monotonic()*1000)) #remove decimal for monotonic time (0.000)
            # Combine the precise timestamp with the monotonic time
            timeid_with_suffix = f'{timeid}{monotonic_suffix}'
            if timeid_with_suffix not in self.generated_ids:
                self.generated_ids.add(timeid_with_suffix)
                break
        #print(f"\nGenerated ID:{timeid_with_suffix}")
        return timeid_with_suffix

    async def RecallRecentMessages_get_documents(adjusted_time_timestamp):
        """Asynchronous function to retrieve a document."""
        try:
            document = await self.conversational_memory_collection.get(
                ids=[adjusted_time_timestamp]
#                where=
                    )
            print(f"\n\n\n\n\n{document['documents']}")
        except Exception:
            return None
        else:
            return document
    async def RecallRecentMessages(self, user_input):
        """
                # Create an event loop
                        loop = asyncio.new_event_loop()
                        # Run the async function within the loop
                        loop.run_until_complete(self.RecallRecentMessages())
                        # Close the loop when finished
                        loop.close()
                asyncio.run(self.RecallRecentMessages(user_input))

                """
        retrieved_documents=[]
        document_count = 0
        current_timestamp = self.generate_timeID() 
        #20231207@164221.000334214 becomes 2023-12-07T16:42:21.000000+00:00
        current_timestampISO = current_timestamp.isoformat(timespec="milliseconds") + ".000"  # millisecond precision
        adjusted_time_timestamp = current_timestampISO
        print(current_timestamp)
        print(current_timestampISO)
        # Loop until five documents are retrieved
        while document_count < 5:
            # Generate document ID in desired format
            # Send asynchronous request to retrieve document
            RecallRecentMessages_get_documents = asyncio.create_task(RecallRecentMessages_get_documents(adjusted_time_timestamp))
            # Move back in time for next iteration
            adjusted_time_timestamp -= timedelta(seconds=1)
            # Get the retrieved document
            document = await RecallRecentMessages_get_documents
            # Add document and update counter if successful
            if document:
                retrieved_documents.append(document)
                document_count += 1
        for document in retrieved_documents:
            print(document['documents'])

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

        query_assistant_results = self.conversational_memory_collection.query(
            query_texts=[user_input],
            where={"role": "assistant"},           #where = {"interlocutor": f"{self.userName}"}
            n_results=15,
        )
        
        #----|FILTER RESULTS| INCREMENTALLY RELAX RESULTS FILTER UNTIL MINIMUM OF 5 ARE KEPT
        # Set your initial threshold
        min_cosinesimilarity = 1.0 # Start with the highest similarity
        min_cosinesimilarity_threshold = 0.55 # stop if less than this similar 0.6=%60
        max_results = 5
        token_limit = 500 # Set your desired token limit
        token_count = 0 # Initialize a token counter
        exit_flag = False
        while True:
            if not query_assistant_results['ids'][0]:
                self.database_results_comparison_set=[]
                RecallMemory_database_results=""
                return RecallMemory_database_results
            max_cosinedistance = 1 - min_cosinesimilarity
            print(f"Including distances below {max_cosinedistance}")  #debugging
            filtered_results = {key: [] for key in query_assistant_results.keys()}            
            database_search_results_list = []

            for i in range(len(query_assistant_results['ids'][0])):   
                if query_assistant_results['distances'][0][i] <= max_cosinedistance:
                    for key in query_assistant_results:
                        if query_assistant_results[key] is not None:
                            if key == 'ids':
                                formatted_id = datetime.strptime(query_assistant_results['ids'][0][i].split('.')[0], "%Y%m%d@%H%M%S").strftime("%Y-%m-%d@%H:%M:%S") 
                                filtered_results['ids'].append(formatted_id)
                            else:                
                                filtered_results[key].append(query_assistant_results[key][0][i])

                    #result_token_count= len(word_tokenize(query_assistant_results['documents'][0][i]))
                    result_token_count= len(word_tokenize(filtered_results['documents'][i]))
                    token_count += result_token_count # Update the token counter
                    print(f"Token count for result {[i]}: {result_token_count} [{token_count}/{token_limit}]")  # Print the token count
                    if token_count >= token_limit:  # Check the token limit
                        print(f"Token count limit met at  {token_count}/{token_limit}")  # Print the token count
                        exit_flag=True
                        break

                    # Break the loop if we have at least 5 results or have checked all results or if no results are within relevancy threshold
                    if len(filtered_results['ids']) >= max_results or min_cosinesimilarity<=min_cosinesimilarity_threshold or min_cosinesimilarity <= 0.0:
                        min_cosinesimilarity_percentage = f"{(min_cosinesimilarity * 100)}%"
                        min_cosinesimilarity_threshold_percentage = f"{(min_cosinesimilarity_threshold * 100)}%"
                        print(f"search ended with results [{len(filtered_results['ids'])}/{max_results}] and relevance [{min_cosinesimilarity_percentage}>{min_cosinesimilarity_threshold_percentage}]")  # Print the token count
                        exit_flag=True
                        break
            # Decrement the threshold by a small amount (e.g., 0.01)
            min_cosinesimilarity -= 0.03
            if exit_flag:
                break
        for i in range(len(filtered_results['ids'])):
            interlocutor_variable = filtered_results['metadatas'][i].get('interlocutor', '')
            rolemetadata = filtered_results['metadatas'][i].get('role', '')
            timestamp=filtered_results['ids'][i]
            entry_text = filtered_results['documents'][i]
            filtered_result_item=f"Historical search result {i+1} timestamp[{timestamp}] speaker[{rolemetadata} {interlocutor_variable}] entry:[{entry_text}]"
            database_search_results_list.append(filtered_result_item)
            
        database_search_results_string = "\n".join(database_search_results_list)
        RecallMemory_database_results = database_search_results_string.replace(r'\s{2,}', ' ').replace('\n\n\n', '\n\n').strip()        
        # Now `filtered_results` only contains the results with a distance less than `max_cosinedistance`

        self.database_results_comparison_set=database_search_results_list

        #print(self.database_results_comparison_set)#this is used in AddMemory to not redundantly add similar information
        RecallMemory_database_results=RecallMemory_database_results
        print(RecallMemory_database_results)
        #|-----|HISTORY/DB RESULTS CHAT TEMPLATE|-----| conveys to the model what it's looking at when viewing db search results.
            #this text can be extremely picky.
        history_template = f"""Database results are additional context items for your response, formatted as Entry# timestamp speaker-title speaker-name: [content text]. Consider only the content, Do not replicate formatting or titles in any way
database results:
{RecallMemory_database_results}"""
        #|-----|ADD DB RESULTS TO SYSTEM MESSAGES|-----|
        global system_template, messages
        for message in messages:
            if message["role"] == "system":
                message["content"] = f'{system_template}{history_template}'
        RecallMemory_database_results=""
        return RecallMemory_database_results

