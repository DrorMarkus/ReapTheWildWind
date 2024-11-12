In this folder (“ReapWildWind_ReplicationMaterials”), we have included, first, 3 data files:
1. MediaDispersionSignals.csv - this file contains the trace and entropy dispersion signals that we generated and used in our experiments. Per our license agreement and copyright regulations, we cannot share the original articles and embeddings. Instead, we provide here the raw signals to be used to replicate the first stage of the experiment (until human validation).  These can also be used for further research directions to delve into the dispersion signals over time.
2. SeedStorms.csv - this file contains the original seed storms we used to initiate the experiment iterations. Together with the dispersion signals, these files should allow for complete replication of the first round of the experiments. The next stages include human validation and therefore may be more difficult to replicate precisely.
3. MediaStorms_1996-2016_withcategories.xlsx - this file contains the complete list of researcher-validated media storms identified in our experiments and contains metadata of the start and end dates, media storm labels, and several human-labelled descriptive categorical labels.


Scripts:


The first stage of our method involves converting a corpus of full-text news articles to embeddings. While we could not share the raw articles and data (as explained above), we include here scripts demonstrating the method used to generate embeddings for two of our textual dimensions:
1. Via LLMs - Stage1_GenerateEmbeddings_llm.py
2. Using entities - Stage1_GenerateEmbeddings_entities.py
The approach shown here is essentially the same for any other representation researchers may wish to utilize.


Next, we provide a script used for generating the dispersion signals based on the embeddings. Our script consists of two stages. First, calculating the trace based on embeddings (we used this approach for our LLM; plot components (NEAT)*; and topic vectors. In the second section, we calculate the entropy based on the entity vectors. We chose entropy both to efficiently handle the large, sparse matrix and reduce memory overhead, and since we were dealing with categorical frequencies as opposed to continuous embeddings.
3. Stage2_Generating Dispersion Signals_.py


Then, we provide the scripts to run the two experiments - the In-Period (Experiment A) and the Out-Period (Experiment B). Each experiment contains two scripts - the random search to ascertain the best anomaly detection configuration using a list of initial seed storms, and a script that outputs a list of media storm candidates and interactive visualizations to aid the human coder. Essentially, these two scripts can be used in multiple iterations until convergence. 


The scripts contain detailed comments explaining each step of the procedures. 


4. Stage3_ExperimentA_part1_clean.py
5. Stage3_ExperimentA_part2_clean.py
6. Stage3_ExperimentB_part1_clean.py
7. Stage3_ExperimentB_part2_clean.py




We welcome you to reach out with any questions or comments to dror.markus@mail.huji.ac.il


*https://aclanthology.org/2022.findings-naacl.133.pdf
