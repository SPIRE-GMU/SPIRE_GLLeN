To build Neo4j database: 
 Edit the destination for the C files in 1_exeBenchToC.py and then run it
 Next run 2_gccToCFG.py which will run gcc on all c files and build CFG's 
 Then run 03_cfg_to_json.py to convert te CFG's to Json for neo4j
 Next run 04_json_to_neo4j.py to import all json files into the Neo4J database
 Then run 5_codeBERT_embeddings.py to build the embeddings for KNN search into the database
 Lastly run 6_KNN_search.py with a CFG file to search the database for the most similiar file. 