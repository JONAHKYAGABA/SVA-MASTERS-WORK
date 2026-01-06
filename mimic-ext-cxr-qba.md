PhysioNet
Share
About
Explore 

Search PhysioNet

kyagabajonah 
 Database  Credentialed Access

MIMIC-Ext-CXR-QBA: A Structured, Tagged, and Localized Visual Question Answering Dataset with Question-Box-Answer Triplets and Scene Graphs for Chest X-ray Images
Philip Müller  ,  Friederike Jungmann  ,  Georgios Kaissis  ,  Daniel Rueckert 

Published: July 22, 2025. Version: 1.0.0

When using this resource, please cite: (show more options)
Müller, P., Jungmann, F., Kaissis, G., & Rueckert, D. (2025). MIMIC-Ext-CXR-QBA: A Structured, Tagged, and Localized Visual Question Answering Dataset with Question-Box-Answer Triplets and Scene Graphs for Chest X-ray Images (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/8qmz-da41

Please include the standard citation for PhysioNet: (show more options)
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

Abstract
Visual Question Answering (VQA) enables flexible and context-dependent analysis of medical images, such as chest X-rays (CXRs), by allowing users to pose specific questions and receive nuanced answers. However, existing CXR VQA datasets are typically limited to short and simplistic answer, lack localization information (such as bounding boxes), and provide little structured metadata (e.g., hierarchical answer formats or tags like region and finding annotations). To address these limitations, we introduce MIMIC-Ext-CXR-QBA, a new large-scale CXR VQA dataset derived from MIMIC-CXR, comprising 42 million QA pairs, which provides multi-granular, hierarchical answers composed of full sentences in the style of radiology reports, as well as detailed bounding boxes, and structured tags. Additionally, we provide scene graphs for each study, containing both regions and observation nodes with bounding boxes, tags, and textual descriptions derived from the original radiology reports. We created the scene graphs using LLM-based information extraction, semantic mention mapping, and localization models before generating question-answer pairs based on the extracted information stored in these graphs. Using automatic quality assessments, we have selected 31,230,906 QA pairs intended for pre-training and 7,532,281 of these intended for fine-tuning VQA models, therefore providing, to the best of our knowledge, the most sophisticated and largest VQA dataset for CXRs yet.

Background
With the emergence of Large Language Models (LLMs) and Large Multimodal Models (LMMs), interactive and conversational tasks have become a common way to interpret medical images, including chest X-rays (CXR) [1-5]. One widely studied interactive task is Visual Question Answering (VQA), where a model is given an image and a corresponding textual question and is expected to generate an answer. Unlike conventional medical imaging approaches that produce fixed outputs—such as classification labels, bounding boxes, segmentation masks, or textual reports—VQA enables user-driven, context-dependent interpretations, allowing for more flexible insights.

VQA allows the formulation of a variety of problem types, ranging from simple yes/no questions to more complex free-text answers. However, training robust VQA models for medical applications necessitates high-quality, large-scale training datasets. Existing CXR VQA datasets [1], [6-9] suffer from several limitations: they often contain only short and simplistic answers, lack localization information (such as bounding boxes), and provide little structured metadata (e.g., hierarchical answer formats, region and finding annotations, or uncertainty estimates). Additionally, their relatively small size constrains their utility for pretraining and necessitates fine-tuning on limited data.

To address these challenges, we introduce a new large-scale CXR VQA dataset derived from MIMIC-CXR [10-12], consisting of 42,172,827 QA pairs. Using automatic quality assessment, we have selected 31,230,906 pairs intended for pre-training and 7,532,281 of these intended for fine-tuning.

Unlike prior datasets, each QA pair includes multi-granular, hierarchical answers composed of full, structured sentences in the style of radiology reports. Furthermore, our dataset provides detailed bounding boxes and additional structured tags (e.g., findings, anatomical regions, probability estimates), enhancing interpretability and facilitating the development of more advanced and transparent VQA models for medical imaging.

Methods
To construct our visual question-answering dataset from MIMIC-CXR [10-12], we employ three key steps: scene graph generation, question-answer generation, and quality assessment.

We first construct scene graphs using both LLM-based information extraction (from the reports) and semantic concept mapping. Extracted observations are then associated with bounding boxes provided by anatomical region localization models. These scene graphs provide a structured description of the study, including sentences (derived from the report) for individual observations. They serve as a data source for our question-answer generation, where we utilize both template-based answers and answers derived from the rewritten report sentences. Finally, we automatically assess the quality of question-answer pairs using LLM-based evaluations.

1. Scene Graph Construction
We construct the scene graphs in three major steps, namely a) region localization, b) information extraction, and c) construction with entity mapping.

Region Localization
The bounding boxes in our scene graphs (and the derived QA-pairs) are based on fine-grained anatomical structures, allowing us to localize associated findings very precisely.

We use the CXAS [13] model to predict segmentation masks of 158 anatomical structures in the chest. We apply CXAS on the 377,110 CXRs from MIMIC-CXR-JPG [12], [14,15] and postprocess the resulting masks using morphological transformations to remove noise. Additionally, we use the bounding boxes provided by the Chest ImaGenome [12], [16] dataset, which are provided for 29 anatomical structures in most frontal images of MIMIC-CXR. Next, we derive a total of 257 localized anatomical structures based on combinations (e.g. intersections, unions, super bounding boxes, etc.) of the available masks and bounding boxes. Finally, we discard any masks or boxes that are too small and derive bounding boxes from the segmentation masks. Note that we define 53 further regions/structures that are either non-localized (e.g. interstitial) or for which we do not have bounding boxes, leading to a total of 310 structures/regions.

Information Extraction
We use the 227,827 free-text radiology reports provided by MIMIC-CXR as the main source of information for our scene graphs. Using the Llama 3.1 70B [17] model with few-shot prompting, we extract the relevant information in three steps.

First, we extract individual sentences from the reports, detect their sections (e.g. FINDINGS, IMPRESSION, INDICATION, …), discard sentences without relevant information, and merge sentences containing similar information (e.g. if findings are described in both the FINDINGS and IMPRESSION section). Therefore, each full report is passed in a single step to the LLM, which predicts the individually separated sentences as well as their sections and related sentences.

Next, we extract information about the INDICATION section and detect which FINDINGS or IMPRESSION sentences may provide information related to the indication. Therefore, the extracted INDICATION sentences and a list of all FINDINGS and IMPRESSION sentences are passed to the LLM, which predicts the following (in a json-structure):

The INDICATION sentences rewritten (cleaned from different formattings in the report) as an indication summary.
Patient info extracted from the INDICATION sentences, typically containing the patient’s sex.
The clinical indication if there is any given in the INDICATION sentences
The expected evaluation (i.e. what should be assessed using the CXR) as named in the INDICATION sentences.
A list of FINDING/IMPRESSION sentences (their IDs) that my be used to answer the indication.
A short answer (”answer_for_indication”) that would be given to the indication / evaluation question, considering what is written in the FINDING/IMPRESSION sentences.
Finally, we extract individual observations described in the FINDING/IMPRESSION sentences. Therefore, we pass each FINDING/IMPRESSION sentence individually to the LLM and let the LLM predict json-objects for individual observation mentioned in the sentence (which may per no, one, or more observations per sentence). Each observation includes the following:

name: short name of the observation derived from the sentence)
summary_sentence: textual description of the observation (derived from the sentence)
entity: the associated finding entities - regions: the associated anatomical regions
probability: is this as positive or negative finding and how likely
temporal, spread, …: modifiers of the finding entities
change, change_sentence: type and description of (longitudinal) change mentioned in the report
children: sub-observations, providing more details (same structure as top-level json)
The LLM is allowed to freely assign values to each of those fields but we provide few-shot examples and guidelines in the prompt, including examples of entities and rules for modifier assignment. For name and summary_sentence, we prompt the model to stay close to the original sentence, but it must remove any mentions of change and only keep the part relevant to the individual observation (if several observations are mentioned in one sentence).

Graph Construction and Entity Mapping
Given the extracted information from the reports and the computed bounding boxes, we now construct the final scene graph. Therefore, we first map individual fields (entity, regions, probability, modifiers, change) to pre-defined sets of values, our reference definitions . This assures high quality and consistency of the scene graphs and enables mapping of observations to bounding boxes. The reference definitions are based on tags used in other datasets (including PadChest [18] and Chest ImaGenome [12], [16]) as well as SNOMED-CT [19] and have been verified by clinical experts. They include synonym lists, hierarchies (categories, …), and relationships. For more robust mapping, we utilize the BioLORD [20] model as a sentence transformer and identify the closest matching concept based on their semantic embeddings.

Next, we simplify region information (merging regions or picking more precise regions) and derive default regions from the finding entities if no regions are given. We merge similar oberservations and check the consistency between observations. We then add pre-defined negative default observations (if no contradicting observations are present) and assemble a graph of observation nodes.

Based on the mentioned regions, we associated bounding boxes with the observations if available. Additionally, we build a tree of all mentioned regions and fill missing intermediate regions based on the reference data. This allows us to build a graph of region nodes relevant to the study.

We construct region-region edges based on the reference data, observation-region edges (located-at) based on the mentioned observation regions, observation-observation edges based on the parent-child structure of observations (and the child-type predicted by the LLM), and observation-sentence relations based the sentences each observation was derived from.

Finally, we attach the indication information extracted from the report. Therefore, we build an additional observation node based on the extracted “answer_for_indication” and the associated finding sentences (and their observations).

2. Question-Answer Generation
We generate question-answer pairs following a template-based approach based on the information available in the scene graphs. However, we try to utilize the observation sentences (derived from the report sentences) wherever possible to provide diverse and fine-grained answers directly derived from the written report sentences.

We structure each answer hierarchically, following the structure of observations, i.e. with multiple individual “top-level” answers and optionally sub-answers. Each of the individual answers contains text (the answer itself), bounding boxes (wherever available), and additional information derived from the observations in the scene graph (regions, findings, modifiers, probability, …). Additionally, we categorize the answer parts into:

main-answers: required to answer the question, there is always at least one main-answer per question.
details: providing additional details for the main answer.
related-information: not directly answering the question, but may be related and provides context.
Main answers are either created template-based or created based on observations in the scene graph. All other types are always derived from observations in the graph.

We utilize different generation strategies to i) identify the observations relevant for the question, ii) fill question and main-answer templates based on the information in the scene graph, and iii) convert the identified observations into answers. We use the following four generation strategies (each with one or more different templates):

“Indication” Strategy
In this strategy, we use the extracted indication (if available) as the question. The answer starts with a main-answer based on the indication observation (i.e. the answer to the indication based on the finding sentences), while detail sub-answers are constructed based on all associated finding observations. We include this question, if an indication observation is present in the scene graph.

“Abnormal“ Strategy
In this strategy, we generate questions about abnormalities. This includes descriptions of the full study or specific categories of observations (e.g. devices), description of only abnormal findings, and yes/no questions of whether there are positive findings (overall or of specific categories) present in the study.

Answers to description questions include all related observations as main answers. For yes/no questions, we first create a template-based main answer and then add the related observations as detail answers. We include each of these questions for most of the scene graphs, but ignore samples where we can’t guarantee the correctness based on the scene graphs.

“Region“ Strategy
In this strategy, we generate question about anatomical regions. This includes describing regions, answering yes/no questions about the abnormality of regions, or describing specific aspects of regions (e.g. devices).

Answers to description questions include all region-related observations as main answers. For yes/no questions, we create a template-based main answer and then the related observations as detail answers. Additionally, we provide “related-information” answers if there are aspects in other regions that might be related (e.g. parent/child regions, other lateralities, …). We always include these questions for a set of default regions (the lungs, the heart, …) and include questions about regions mentioned in observations, as well as their parent regions. Additionally, we randomly sample regions to ask about. Their sampling probabilities are computed based on how often they are associated with positive vs. negative findings, i.e. the more often a region is associated with positive findings and the less often it is associated with negative findings, the more often we sample it as a question. This assures that we generate additional “negative” questions for regions that are only/mostly mentioned with positive findings.

“Finding “ Strategy
In this strategy, we generate question about specific findings (radiological findings, diseaes, devices, …). This includes descriptions of findings, yes/no questions about the presence of findings, location of findings, and severity of findings.

Answers to description questions include all finding-related observations as main answers. For yes/no, location, and severity questions, we create a template-based main answer and add the related observations as detail answers. Additionally, we provide “related-information” answers if there are aspects that might be related (e.g. other findings in the same region, related findings). We always include these questions for a set of default findings and include questions about findings mentioned in observations (positive or negative), as well as their parent findings. Additionally, we randomly sample findings to ask about. Their sampling probabilities are computed based on how often they are mentioned positively vs. negatively (over all scene graphs), i.e. the more often a finding is mentioned positively and the less often it is mentioned negatively, the more often we sample it as a question. This assures that we generate additional “negative” questions for findings that are only/mostly mentioned positively.

3. Evaluation
We provide two types of quality evaluations for our dataset:

(i) Automatic quality assessment and grading (described in Section 3a)
(ii) Quantitative validation against expert annotations (described in Section 3b)
The automatic quality assessment and grading (i) is provided for every single QA-pair (sample) in the dataset and allows filtering the dataset by different quality criteria or grades, e.g. selecting samples for pre-training or fine-tuning (see below). This assessment is conducted completely automatically by tracking the extraction process (using rules) or using an LLM as a judge. The assessment criteria have been carefully designed and the process has been overseen by two trained radiologists.

Additionally, we provide a quantitative validation of our dataset against expert annotations (ii). More precisely, we conducted an analysis on a subset of our dataset, comparing finding entity tags and bounding boxes from our scene graphs to several hand-labeled expert annotations (publicly available for subsets of MIMIC-CXR). This evaluation assesses the correctness of the scene graph annotations, including the identification of radiological findings / diseases / devices and the localization of corresponding regions. This in turn validates the quality of the QA-pairs derived from these annotations, as answer texts are either template-based (using the tags from the scene graphs) or directly derived from report sentences, while all answer tags and bounding boxes are copied from the scene graph data. For more details on this evaluation, we refer to Section 3b.

3a. Automatic Quality Assessment and Grading
Automatic Scene Graph Quality Assessment
We asses the scene graph extraction quality using simple rules and tracking of issues during the extraction, mapping, and graph construction process:

Criterion	Options (resulting max-rating)
region_quality	
NO_REGIONS (B)
DEFAULT_REGIONS_ONLY (B)
CONTAINS_DEFAULT_REGIONS (A)
CONTAINS_NON_RESOLVED_REGIONS (A)
RESOLVED_REGIONS_ONLY (A++)
entity_quality	
NO_ENTITIES (B)
CONTAINS_NON_RESOLVED_ENTITIES (A)
RESOLVED_ENTITIES_ONLY (A++)
sentence_name_quality	
CHANGE_IN_SENTENCE_OR_NAME (B)
UNDERSCORES_IN_SENTENCE_OR_NAME (A)
NO_ISSUES (A++)
change_quality	
CHANGE_SENTENCE_REMOVED (B)
UNDERSCORES_IN_CHANGE_SENTENCE (A)
CONTAINS_NON_RESOLVED_CHANGES (A)
NO_ISSUES (A++)
issue_level	
DISCARDED (D)
NON_INTERPRETABLE (C)
MOSTLY_INTERPRETABLE (B)
IGNORABLE (A)
FIXABLE (A+)
NO_ISSUES (A++)
localization_quality	
NO_LOCALIZATION (B)
FALLBACK_LOCALIZATION (B)
INCOMPLETE_LOCALIZATION (A)
BBOX_LOCALIZATION (A++)
BBOX_AND_MASK_LOCALIZATION (A++)
Automatic Question-Answer Quality Assessment
We evaluate the question-answer quality using Llama 3.1 8B with few-shot prompting and the following evaluation criteria (each criteria is evaluated independently):

Criterion	Evaluation Level	Context	Options (rating)
Entailment	Sub-answer	
Report
Question
All sub-answers
ALIGNED_MENTIONED (A++)
ALIGNED_INFERABLE (A++)
ALIGNED_NEGATIVE_NOT_MENTIONED (A)
ALIGNED_GENERAL_STATEMENT (A)
NON_ALIGNED_NON_INFERABLE (B)
NON_ALIGNED_MISLEADING (C)
NON_ALIGNED_CONTRADICTING (D)
Relevance	Sub-answer	
Question
All sub-answers
RELEVANT_MAIN_ANSWER (A++)
RELEVANT_MAIN_ANSWER for related info (A)
RELATED_INFO (A++)
RELATED_INFO for main answer or details (A+) REDUNDANT_INFO (A)
IRRELEVANT_INFO (A)
Completeness	Full answer	
Question
Full answer (all sub-answers)
FULLY_COMPLETE (A++)
DETAILS_MISSING (A+)
NOT_ANSWERED (B)
INCOMPLETE_NON_MISLEADING (B)
INCOMPLETE_MISLEADING (C)
Question clarity	Question	
Question
OPTIMAL (A++)
UNUSUAL_SENTENCE_STRUCTURE (A)
GRAMMATICAL_ERRORS (A)
UNCLEAR_QUESTION (B)
UNRELATED_TO_CHEST_XRAY (B)
UNANSWERABLE (C)
Answer clarity	Sub-answer	
All sub-answers
OPTIMAL (A++)
UNUSUAL_SENTENCE_STRUCTURE (A)
GRAMMATICAL_ERRORS (A)
UNCLEAR_ANSWER (B)
NOT_UNDERSTANDABLE (C)
Final Quality Grading
Based on the QA quality and extraction quality, we compute an overall rating for each QA-pair, considering the minimum rating of the full scene graph and all answers of the current question in this sample.

Based on these rating we prepare two main datasets recommended for training:

Pre-training grade: everything with grade B or better
Fine-tuning grade: everything with grade A or better
In these datasets, we also exclude all non-frontal images (because the bounding box quality in generally low in these cases) and remove all studies without any frontal images. Note that we also exclude samples, where the evaluation failed (mainly due to issues in the LLM-based evaluation). These samples (almost 20% of all samples) are not necessarily of bad quality, but we cannot guarantee the quality and therefore do not recommend using them for training. Using larger evaluation models may reduce the number of non-validated samples, but we decided to not further optimize this, as there is already a large and diverse set of rated samples. While we recommend using one of the two datasets, we also release the full dataset including non-validated and lower-grade samples as well as non-frontal images.

3b. Quantitative Validation against Expert Annotations
While the automatic quality assessment in Section 3a provides grades for each QA-pair, which can be useful for filtering, it does not evaluate the correctness of finding/region tags. To address this, we conducted an analysis on a subset of our dataset, comparing finding entity tags and bounding boxes to hand-labeled expert annotations (available for subsets of MIMIC-CXR). We use Chest ImaGenome's scene graphs as a baseline for comparison.

First, we evaluate the plausibility of finding tags by comparing study-level labels derived from our scene graphs to two reference annotation sets: the radiologist annotations in MIMIC-CXR-JPG [12], [14, 15] v.2.1.0 with 13 CheXpert (CXP) [21] classes and the CXR-LT 2024 [12], [22, 23] gold-standard dataset (task 2 test set) with 12 additional rare, long-tail (LT) classes. Our approach (slightly) outperforms Chest ImaGenome, with strong improvements (20%) on long-tail classes, demonstrating the value of our fine-grained finding tags (237 classes) in capturing nuanced study details:

Validation of finding tags against the MIMIC-CXR-JPG Test annotations, using the Matthews Correlation Coefficient (MCC) metric.
Classes	CXP-5	CXP-7	CXP-13	Micro
Ours (scene graphs)	0.80	0.81	0.69	0.71
Chest ImaGenome	0.78	0.80	0.66	0.67
Validation of finding tags against the CXR-LT 2024 Gold annotations, using the Matthews Correlation Coefficient (MCC) metric.
Classes	CXP-7	CXP-13	LT-only	CXR-LT	Micro
Ours (scene graphs)	0.65	0.57	0.71	0.64	0.67
Chest ImaGenome	0.65	0.56	0.59	0.58	0.64
To evaluate the accuracy of finding bounding boxes, we compare them with annotations from MS-CXR [12], [24, 25] (on 6 of the 8 classes with positive samples on all datasets) and REFLACX [12], [26, 27] (on 18 of the 29 classes with positive samples on all datasets). We compute study-level pixel masks for each finding as the union of all bounding boxes from positive observation nodes that contain the specific finding tag.

We calculate pixel-level Intersection-over-Union (IoU), Intersection-over-Prediction (IoP), and Intersection-over-Target (IoT) for each finding class, considering only image pairs with positive predictions and targets. Thresholding at 30% IoU/IoP/IoT, we micro-average the results, reporting the percentage of accurately localized finding-boxes.

On the IoU metric, our scene graphs perform slightly better than the ones from Chest ImaGenome.

The low IoP values indicate that bounding boxes are often too large, but high IoT values suggest that they generally cover the finding boxes well. This discrepancy arises because bounding boxes are derived from anatomical regions mentioned in reports, whereas hand-labeled annotations are more precise. Notably, our approach produces more precise boxes (higher IoP) than Chest ImaGenome, likely due to our large number of fine-grained region annotations (311 region classes).

Validation of finding bounding boxes against MS-CXR, using the pixel-level Intersection-over-Union (IoU), Intersection-over-Prediction (IoP), and
Intersection-over-Target (IoT), each thresholded at 30%, and micro-averaged.
Metric	IoU@30	IoP@30	IoT@30
Ours (scene graphs)	0.51	0.56	0.94
Chest ImaGenome	0.45	0.48	0.98
Validation of finding bounding boxes against REFLACX, using the pixel-level Intersection-over-Union (IoU), Intersection-over-Prediction (IoP), and
Intersection-over-Target (IoT), each thresholded at 30%, and micro-averaged.
Metric	IoU@30	IoP@30	IoT@30
Ours (scene graphs)	0.45	0.54	0.87
Chest ImaGenome	0.42	0.46	0.95
Data Description
Dataset Statistics
Number of samples and their quality:

Train	Val	Test	Total
# Patients	64,524	500	293	65,317
# Studies	222,180	1,805	3,254	227,239
# QA pairs	41,239,042	600,763	333,022	42,172,827
→ Fine-tuning grade	7,378,344	58,486	95,451	7,532,281
→ Pre-training grade	30,542,190	246,233	442,483	31,230,906
→ Rating A++	1,338,959	10,267	18,775	1,368,001
→ Rating A+	1,092,771	8,610	14,408	1,115,789
→ Rating A	5,237,408	41,758	68,414	5,347,580
→ Rating B	24,241,667	197,103	373,911	24,812,681
→ Rating C	683,310	5,848	11,589	700,747
→ Rating D	534,169	4,326	9,268	547,763
→ Unrated	8,110,758	65,110	104,398	8,280,266
# Sub-answers	88,876,344	717,117	1,321,739	90,915,200
Number of questions for each question type and strategy:

Question Strategy (identifier)	Question Type (identifier)	Train	Val	Test	Total
Indication (indication)	Indication (A_indication)	213,506	1,741	3,047	218,294
Abnormal (abnormal)		11,652,463	94,095	168,826	11,915,384
describe_all (B01_describe_all)	203,868	1,644	2,921	208,433
describe_abnormal (B02_describe_abnormal)	203,868	1,644	2,921	208,433
is_abnormal (B03_is_abnormal)	203,868	1,644	2,921	208,433
is_normal (B04_is_normal)	203,868	1,644	2,921	208,433
describe_subcat (B08_describe_subcat)	2,307,749	18,652	33,928	2,360,329
describe_abnormal_subcat (B09_describe_abnormal_subcat)	2,307,749	18,652	33,928	2,360,329
is_abnormal_subcat (B10_is_abnormal_subcat)	2,307,749	18,652	33,928	2,360,329
is_normal_subcat (B11_is_normal_subcat)	1,630,944	13,152	23,368	1,667,464
describe_device (B12_describe_device)	934,111	7,534	13,014	954,659
has_devices (B13_has_devices)	934,111	7,534	13,014	954,659
describe_acquisition (B14_describe_acquisition)	6,842	55	120	7,017
describe_imaging_artifacts (B15_describe_imaging_artifacts)	203,868	1,644	2,921	208,433
has_imaging_artifacts (B16_has_imaging_artifacts)	203,868	1,644	2,921	208,433
Region (region_abnormal)		20,169,684	162,217	288,807	20,620,708
describe_region (C01_describe_region)	2,768,270	22,031	36,846	2,827,147
describe_abnormal_region (C02_describe_abnormal_region)	2,768,278	21,889	37,010	2,827,177
is_abnormal_region (C03_is_abnormal_region)	2,773,041	22,049	36,976	2,832,066
is_normal_region (C04_is_normal_region)	2,772,856	21,924	37,058	2,831,838
describe_region_device (C07_describe_region_device)	4,543,059	37021	70,203	4,650,283
has_region_device (C08_has_region_device)	4,544,180	37,303	70,714	4,652,197
Finding (finding)		9,203,389	74,969	140,083	9,418,441
describe_finding (D01_describe_finding)	2,491,473	20,352	38,273	2,550,098
has_finding (D02_has_finding)	2,492,466	20,395	38,355	2,551,216
where_is_finding (D03_where_is_finding)	1,975,568	15,966	29,618	2,021,152
how_severe_is_finding (D04_how_severe_is_finding)	1,297,647	10,364	17,602	1,325,613
describe_device (D05_describe_device)	318,476	2,706	5,446	326,628
has_device (D06_has_device)	313,717	2,587	5,374	321,678
where_is_device (D07_where_is_device)	314,042	2,599	5,415	322,056
Files and Structure
Directory Structure
├── metadata (2.3 GB)
│    ├── patient_metadata.csv.gz
│    ├── study_metadata.csv.gz
│    ├── image_metadata.csv.gz
│    ├── question_metadata.csv.gz
│    ├── question_image_metadata.csv.gz
│    ├── answer_metadata.csv.gz
│    ├── answer_image_metadata.csv.gz
│    └── dataset_info.json
├── stats (4.5 GB)
│    └── ...
├── scene_data.zip (1.3 GB)
├── qa.zip (7.5 GB)
├── exports (12.4 GB)
│    └── ...
└── quality_mappings.csv (5 KB)
Metadata (”metadata” dir)
We provide metadata for all scene graphs and question-answer pairs in the metadata directory. The metadata is provided on different levels (patient, study, image, question, question-image, answer, and answer-image) with according number of rows. Each of the metadata files is provided in two redundant versions:

.csv.gz (compressed csv): for easy interpretation and
.parquet: for fast reading
These metadata files can be used to filter the dataset on different levels (patient, study, question, …) by different criteria. Therefore, each file comes with unique IDs and additional metadata that may be relevant for that level. An overview is provided below:

Metadata file	1 row per	Index columns	Example metadata	Total # Rows
patient_metadata	patient	patient_id	
total_studies (of for this patient)
total_study_timespan (interval of study timestamps)
65,317
study_metadata	study	patient_id, study_id	
quality (ratings)
num_observations (in the scene graph)
procedure (from DICOM metadata)
timestamp_since_first (relative to first study of patient)
timespan_since_prev (relative to previous study)
227,239
image_metadata	image	patient_id, study_id, image_id	
view_position (from DICOM metadata)
patient_orientation (from DICOM metadata)
size
localization_quality
376,175
question_metadata	question	patient_id, study_id, question_id	
quality (ratings)
question_type
question_strategy
contains_report_answers (are any answers derived from report sentences)
contains_template_answers (are any answers based on templates but not directly from sentences)
num_answers
42,172,827
question_image_metadata	question-image pair	patient_id, study_id, question_id, image_id	
localization_quality
70,045,778
answer_metadata	answer	patient_id, study_id, question_id, answer_id	
quality (ratings)
answer_type (main answer, details or related information)
answer_level (hierarchy level of sub-answers)
from_report (whether it was derived from a report sentence)
90,915,200
answer_image_metadata	answer-image pair	patient_id, study_id, question_id, answer_id, image_id	
num_regions
localization_quality
151,539,450
Additionally, the dataset_info.json describes the sets of possible values for different tags of answers/observations, i.e. possible finding entity names, region names, finding categories and subcategories, answer types, modifiers, etc.

Statistics (”stats” dir)
We provide additional information and statistics about scene graphs and question-answer pairs in the stats directory. This include aggregate statistics as well as observation-level (for scene graphs) or answer-level (for questions) information. It may for example be used for more advanced data filtering or to compute dataset characteristics without having to load individual scene-graphs or qa-samples (which would be much more expansive).

Aggregate statistics about scene graphs are named as study*.csv and include (among others) the percentages of positive/negative observations for different regions, entities, and categories.

Observation-level information for scene-graphs are named as all_obs*.csv and include (among others) information about positive/negative observations, entities, regions, categories.

(Sub-)answer-level information for qa-samples are named as all_ans*.csv and include (among others) information about positive/negative answers, entities, regions, categories.

Scene Graph Format (”scene_data.zip”)
All scene graphs (and related metadata) can be found in the scene_data.zip file, which contains a folder structure in the following format:

p1x/p1xxxxxxx/sxxxxxxxx.scene_graph.json
p1x/p1xxxxxxx/sxxxxxxxx.metadata.json
where p1x refers to the first 2 digits of the subject_id, p1xxxxxxx to the full subject_id, and sxxxxxxxx to the full study_id.

The sxxxxxxxx.metadata.json file contains study metadata as also provided in the study_metadata.csv.gz file.

The sxxxxxxxx.scene_graph.json file contains the scene graph in the following format:

{
  "patient_id": "p1xxxxxxx",  // see metadata
  "study_id": "sxxxxxxxx",  // see metadata
  // original report sentences (= sentence nodes of scene graph)
  "sentences": {
    "S01": {
      "sent_id": "S01",
      "section": "FINDINGS", 
      "section_type": "FINDINGS",
      "sentence": "No new focal consolidation."
    }, 
    ...  // more sentences
  }, 
  // keys for "observations"
  "top_level_obs_ids": ["O01", "O02", ...],  
  // observations in the study (= observation nodes of scene graph)
  "observations": {
    "O01": {
      "obs_id": "O01",
      "name": "no focal consolidation",
      "summary_sentence": "There is no focal consolidation.", 
      "child_type": null, 
      "child_level": 0, 
      "regions": [{"region": "lungs", "distances": []}], 
      "non_resolved_regions": [], 
      "laterality": "bilateral", 
      "default_regions": ["lungs"],
      "obs_entities": ["consolidation"],
      "obs_entities_parents": [], 
      "non_resolved_obs_entities": [], 
      "obs_categories": ["ANATOMICAL_FINDING", "DISEASE"],
      "obs_subcategories": ["LUNG_FIELD", "PULMONARY_DISEASES", "INFECTION"],
      "probability": "negative", 
      "certainty": "certain", 
      "positiveness": "neg",
      "modifiers": {"temporal": [],"severity": [], "texture": [], "spread": ["focal"]},
      "changes": ["no new"],
      "change_sentence": "No new focal consolidation is visible.", 
      "from_report": true,  // derived from report sentences or template-based?
      "obs_quality": {...},
      "localization": {
	      // one item for each image
	      "[image_id]": {
		      "image_id": "[image_id]",
		      "localization_reference_ids": ["lungs"],
		      // list of bboxes in (x_1, y_1, x_2, y_2) format in pixel cooridnates
		      "bboxes": [[888.0, 370.0, 1610.0, 1642.0 ],   
		                 [136.0, 402.0, 898.0, 1678.0  ]],
		      "instance_mask_ids": ["lungs"],
		      "missing_localization": [],
		      "is_fallback": false,  
		      "localization_quality": ... 
		    },
		  },
    },
    ...  // more observations
	},
	// information related to indication section of report
	"indication": {
	   "indication_summary": "Female with HIV, experiencing chest pain and dyspnea; should be evaluated for infiltrate and effusion.", 
	   "patient_info": "Female, HIV-positive, with chest pain and dyspnea.",
	   "indication": "Chest pain and dyspnea.", 
	   "evaluation": "Evaluate for infiltrate and effusion.",
	   "associated_sentence_ids": ["S05", ...],
	   "associated_obs_ids": ["O03", ...],
	   "answer_for_indication": {
	     // this has the same form as an observation node in "observations"
		   "obs_id": "OIND",  // this ID is always the same
		   "name": "...",
		   ...
		 }
	},
	// regions relevant for the study (= region nodes of scene graph)
	"regions": {
    "left lung": {
      "region": "left lung",
      "laterality": "left",
      "localization": { ... }  // same format as for observation nodes
      "region_localization_quality": ...
    },
    ...  // more regions
  },
  // relations between observation and region nodes
  "located_at_relations": [
    {"region": "lungs", "observation_id": "O01", "distances": [], "where_specified": "direct"},
    ... // more relations
  ],
  // relations between observation node pairs
  "obs_relations": [
    {"parent_observation_id": "O02", "child_observation_id":"O02.01", "child_type":"associated_with"},
    ... // more relations
  ],
  // relations between observation and sentence nodes
  "obs_sent_relations": [
    {"observation_id": "O01", "sentence_id": "S01"},
    ... // more relations
  ]
  // relations between region node pairs
	"region_region_relations": [
	  {"region": "lungs", "related_region": "left lung", "relation_type": "sub_region"},
	  {"region": "left lung", "related_region": "right lung", "relation_type": "right"},
	  ... // more relations
	]
	// quality levels for different aspects (larger = better)
	"study_quality": {
    ...
  },
  // localization quality level per imge-id (larger = better)
  "study_img_localization_quality": {
    ...
  }
}
Question-Answer Format (”qa.zip”)
All question-answer data can be found in the qa.zip file, which contains a folder structure in the following format:

p1x/p1xxxxxxx/sxxxxxxxx.qa.json
where p1x refers to the first 2 digits of the subject_id, p1xxxxxxx to the full subject_id, and sxxxxxxxx to the full study_id.

Each of the sxxxxxxxx.qa.json files contains all question-answer pairs (and additional tags) for a single study in the following format:

{
	"patient_id": "p1xxxxxxx",  // see metadata
  "study_id": "sxxxxxxxx",  // see metadata
	"questions": [
	// -> one object per question-answer pair
	{
		"question_id": "xxxxxxxxxxxx",  // see metadata
		"question_type": "describe_all",  // template used for generation
		"question_strategy": "abnormal",  // strategy used for generation
		"variables": { ... },  // template variables used for generation
		"obs_ids":["O01", ...],  // observations (from scene graph) used in answer
		"contains_report_answers": true/false,  // any answers derived from report sentences?
		"contains_template_answers": true/false,  // any answers based on templates but not directly from sentences?
		"extraction_quality": { ... },
		"question_img_localization_quality": { ... },
		"question": "Describe the given study.",
		// list of sub-answers (top-level answers with their sub-answers)
		"answers":[
		  {
		    "answer_id": "xxxxxxxxxxxx",  // see metadata
		    "answer_type":"main_answer",  // main_answer, details, or related_information
		    "answer_level": 0,  // 0 for top-level, >0 for each child-level
		    "text": "There is no focal consolidation.",  // this is the answer text
		    "name_tag": "No focal consolidation",  // summary name of this sub-answer
		    "laterality": "bilateral",
		    "regions": ["lungs"],
		    "obs_entities": ["consolidation"],
		    "obs_entities_parents": [],
		    "obs_categories": ["ANATOMICAL_FINDING", "DISEASE"], 
		    "obs_subcategories": ["LUNG_FIELD", "PULMONARY_DISEASES", "INFECTION"],
		    "certainty": "certain",
		    "positiveness": "neg",
		    // list of modifiers (tuples of modifier type and value)
		    "modifiers": [["spread", "focal"]], 
		    "localization": {
		      // one item for each image
		      "[image_id]": {
			      "image_id": "[image_id]",
			      "localization_reference_ids": ["lungs"],
			      // list of bboxes in (x_1, y_1, x_2, y_2) format in pixel cooridnates
			      "bboxes": [[888.0, 370.0, 1610.0, 1642.0 ],   
			                 [136.0, 402.0, 898.0, 1678.0  ]],
			      "instance_mask_ids": ["lungs"],
			      "missing_localization": [],
			      "is_fallback": false,  
			      "localization_quality": ... 
			    },
			  },
			  // contain child-answers if there are any (same object format as top-level answer)
		    "sub_answers": [],
		    "from_report": true/false,  // derived from report sentences?
		    "extraction_quality": {...},
		    "answer_quality": {...},
		  },
		  ... // more top-level answers
		],
		"question_quality": {...}
  },
  ... // more questions
]} 
		    
Exports (”exports” dir)
Here we provide subsets of the full dataset.

We have two full copies of the dataset:

A_frontal(Fine-tuning grade): Only questions with a quality rating of A, A+, or A++, and only frontal images (7,532,281 QA-pairs, 3 GB). We recommend this dataset for fine-tuning / instruction-tuning purposes.
B_frontal (Pre-training grade): Only questions with a quality rating of B, A, A+, or A++, and only frontal images (31,230,906 QA-pairs, 9.4 GB). We recommend this dataset for pre-training purposes. (this is a superset of A_frontal)
Each of these datasets contains the metadata dir, scene_graph.zip and qa.zip.

Additionally, we provide filtered metadata files for further subsets of these. They are provided as sub-folders in the metadata dirs. We provide the following:

A_frontal/metadata/Ap: Quality rating of A+ or A++ (2,389,739 QA-pairs)
A_frontal/metadata/App: Quality rating of A++ (1,318,885 QA-pairs)
A_frontal/metadata/q1M: random 1M question subset with A, A+, or A++ ratings (1M QA-pairs)
A_frontal/metadata/Ap_q1M: random 1M question subset with A+ or A++ ratings (1M QA-pairs)
A_frontal/metadata/App_q1M: random 1M question subset with A++ ratings (1M QA-pairs)
B_frontal/metadata/q1M: random 1M question subset with B, A, A+, or A++ ratings (1M QA-pairs)
Quality mappings (”quality_mappings.csv”)
This file provides mappings from the raw encodings of quality values (fields in the JSON-files or columns in the metadata files) to their respective fields.

Usage Notes
Dataset Utility
Fine-grained finding and pathology classification
Pathology localization
Fine-grained longitudinal analysis
Structured and grounded radiology report generation
Structured, grounded, and localized visual question answering
Further derived datasets and tasks
Loading and Filtering the Data
All data can be loaded directly from the provided files using standard utilities (e.g. pandas, unzip, json-loaders).

A simple approach would be the following:

Select the relevant samples by loading, merging, and filtering the metadata files (e.g. using pandas). Also consider the quality_mappings.csv file when filtering based on quality ratings.
Load the relevant scene-graph / QA files per study (e.g. by first extracting the zip files using unzip and then loading the json files)
Known Limitations
Our template-based questions and answers may limit variability and introduce grammatical errors, though this is mitigated by deriving some answers directly from report sentences and by our quality assessment.

Additionally, our approach focuses on individual studies and we do not include longitudinal or differential questions.

Finally, our dataset is a silver dataset—constructed using models, rules, and templates—without human annotations. As a result, it may contain errors introduced by these models or our approach and should be interpreted with caution.

Comparison with Other Datasets
Comparison with Scene Graphs Datasets
Compared to other scene graph datasets derived from MIMIC-CXR, our scene graphs provide bounding boxes for both observation and regions nodes, rewritten sentences for each observation as well as fine-grained finding and region classes.

Ours	Chest ImaGenome	RadGraph
Bounding boxes	regions and observations (derived from fine-grained anatomical structures)	only for regions	none
# Finding classes	221	53	no mapping
# Region classes	310 (257 with bboxes)	29	no mapping
Rewritten sentences per observation	yes	no	no (but text spans are provided)
Hierarchical observations	yes	no	no (but relationships provided)
Longitudinal	no	yes	no
Extraction method	Segmentation model + LLM + semantic concept matching	Object detector + rule-based	Relation extraction model
Comparison with VQA Datasets
Compared to existing VQA datasets, our dataset provides three key benefits: a) it provides structured full sentence answers instead of simple short answers, b) it provides bounding boxes for each (sub-)answer, and c) it is much larger than (most) other datasets. Note, however, that some existing VQA dataset use clinical annotators and may thus provide more reliable answers.

Ours	VQA-RAD	SLAKE	MIMIC-Ext-MIMIC-CXR-VQA	Medical-CXR-VQA dataset	CheXinstruct
Bounding boxes	yes, per sub-answer	no	no	no	no	no
Answer structure	multi-granular, hierarchical, full sentence answers with additional tags	short answers (no full sentences)	short answers (no full sentences)	short answers (no full sentences)	short answers (no full sentences)	short answers (no full sentences)
# QA-Pairs	32.6M	3.5K (includes non-CXR)	14K (includes non-CXR)	377K	780K	8.5M
Question construction	template-based	clinical annotators	clinical annotators	template-based (+ LLM-paraphrasing)	template-based	template-based
Answer construction	mixture of:
template-based
LLM-processed report sentences
clinical annotators	clinical annotators	template-based (+ LLM-paraphrasing)	template-based	template-based
Answer data source	Our scene graphs (LLM + concept matching)	clinical annotators	clinical annotators	Chest ImaGenome	LLM-based extraction from reports	image Annotations (depending on source dataset)
Comparison with Grounded Radiology Report Datasets
Some other datasets provide grounded CXR descriptions. However, these only provide study-level descriptions instead of question-specific answers and bboxes, such that they cannot be used for QA tasks. Also, they do not provide structured class/tag annotations.

Ours	MedTrinityMedTrinity-25M	MAIRA-2 Dataset
Bounding boxes	per sub-answer (question-specific)	ROIs (1 or few per study)	per observation
Individual questions	yes	no	no
Class/tag mappings	findings, regions, …	no	no
Annotation scheme	LLM-based, conditioned on original report + automatic quality control	MLLM-based, partially conditioned on original report	unknown
Ethics
The dataset is a derivative dataset of MIMIC-CXR and thus no new patient data was collected. The ethics approval of the dataset follows from that of the parent MIMIC dataset.

We confirm that all data processing, generation, and training were conducted entirely within a local and secure environment, ensuring data safety and privacy. This includes all usage of LLMs, localization, and embedding models, as well as all (vision-language) model training and evaluation. No data was sent to external APIs or processed by any third-party services.

Acknowledgements
This work was partially funded by ERC Grant Deep4MI (Grant No. 884622).

Conflicts of Interest
The authors have no conflicts of interest to declare.

References
Chen Z, Varma M, Xu J, Paschali M, Veen DV, Johnston A, et al (2024). "A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation". arXiv preprint. arXiv:2401.12208v2.
Tu T, Azizi S, Driess D, Schaekermann M, Amin M, Chang PC, et al (2024). "Towards generalist biomedical AI". NEJM AI. 1(3). doi:10.1056/AIoa2300138.
Müller P, Kaissis G, Rueckert D (2024). "ChEX: Interactive Localization and Region Description in Chest X-rays". In: Leonardis A, Ricci E, Roth S, Russakovsky O, Sattler T, Varol G, editors. Computer Vision – ECCV 2024. Cham: Springer Nature Switzerland. doi:10.1007/978-3-031-72664-4_6.
Lee S, Youn J, Kim H, Kim M, Yoon SH (2025). "CXR-LLAVA: a multimodal large language model for interpreting chest X-ray images". Eur Radiol. doi:10.1007/s00330-024-11339-6.
Shaaban MA, Khan A, Yaqub M (2024). "MedPromptX: Grounded Multimodal Prompting for Chest X-ray Diagnosis". arXiv preprint. arXiv:2403.15585.
Lau JJ, Gayen S, Ben Abacha A, Demner-Fushman D (2018). "A dataset of clinically generated visual questions and answers about radiology images". Sci Data. 5(1). doi:10.1038/sdata.2018.251.
Liu B, Zhan LM, Xu L, Ma L, Yang Y, Wu XM (2021). "Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering". In: 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI). doi: 10.1109/ISBI48211.2021.9434010.
Bae S, Kyung D, Ryu J, Cho E, Lee G, Kweon S, et al (2024). "EHRXQA: A multi-modal question answering dataset for electronic health records with chest x-ray images". In: Proceedings of the 37th International Conference on Neural Information Processing Systems.
Hu X, Gu L, Kobayashi K, Liu L, Zhang M, Harada T, et al (2024) "Interpretable medical image visual question answering via multi-modal relationship graph learning". Medical Image Analysis. 97:103279. doi:10.1016/j.media.2024.103279.
Johnson A, Pollard T, Mark R, Berkowitz S, Horng S (2024). "MIMIC-CXR Database (version 2.1.0)". PhysioNet. doi:10.13026/4jqj-jw95.
Johnson AEW, Pollard TJ, Berkowitz SJ, Greenbaum NR, Lungren MP, Deng C, et al (2019). "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports". Sci Data. 6(1):317. doi:10.1038/s41597-019-0322-0.
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, et al (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals". Circulation. 101(23). doi:10.1161/01.CIR.101.23.e215.
Seibold C, Jaus A, Fink MA, Kim M, Reiß S, Herrmann K, et al (2023). "Accurate fine-grained segmentation of human anatomy in radiographs via volumetric pseudo-labeling". arXiv preprint. arXiv:2306.03934.
Johnson A, Lungren M, Peng Y, Lu Z, Mark R, Berkowitz S, et al (2024). "MIMIC-CXR-JPG - chest radiographs with structured labels (version 2.1.0)". PhysioNet. doi:10.13026/jsn5-t979.
Johnson AEW, Pollard TJ, Greenbaum NR, Lungren MP, Deng C, Peng Y, et al (2019). "MIMIC-CXR: A large publicly available database of labeled chest radiographs". arXiv preprint. arXiv:1901.07042.
Wu J, Agu N, Lourentzou I, Sharma A, Paguio J, Yao JS, et al (2021). "Chest ImaGenome Dataset (version 1.0.0)". PhysioNet. doi:10.13026/wv01-y230.
Grattafiori A, Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, et al (2024). "The llama 3 herd of models". arXiv preprint. arXiv:2407.21783.
Bustos A, Pertusa A, Salinas JM, de la Iglesia-Vayá M (2020). "Padchest: A large chest x-ray image dataset with multi-label annotated reports". Medical image analysis. 66:101797. doi:10.1016/j.media.2020.101797.
SNOMED International (2023). "SNOMED CT". https://www.snomed.org.
Remy F, Demuynck K, Demeester T (2024). "BioLORD-2023: semantic textual representations fusing large language models and clinical knowledge graph insights". Journal of the American Medical Informatics Association. 31(9):1844-55. doi:10.1093/jamia/ocae029.
Irvin J, Rajpurkar P, Ko M, Yu Y, Ciurea-Ilcus S, Chute C, et al (2019). "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison". Proceedings of the AAAI Conference on Artificial Intelligence. 33(01):590–7. doi:10.1609/aaai.v33i01.3301590.
Holste G, Lin M, Wang S, Zhou Y, Wei Y, Chen H, et al (2025). "CXR-LT: Multi-Label Long-Tailed Classification on Chest X-Rays". PhysioNet. doi:10.13026/RYJ9-X506.
Holste G, Zhou Y, Wang S, Jaiswal A, Lin M, Zhuge S, et al (2024). "Towards long-tailed, multi-label disease classification from chest X-ray: Overview of the CXR-LT challenge". Medical Image Analysis. 97:103224. doi:10.1016/j.media.2024.103224.
Boecking B, Usuyama N, Bannur S, Castro DC, Schwaighofer A, Hyland S, et al (2024). "MS-CXR: Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing". PhysioNet. doi:10.13026/9G2Z-JG61.
Boecking B, Usuyama N, Bannur S, Castro DC, Schwaighofer A, Hyland S, et al (2022). "Making the Most of Text Semantics to Improve Biomedical Vision–Language Processing". In: Avidan S, Brostow G, Cissé M, Farinella GM, Hassner T, editors. Computer Vision – ECCV 2022. Cham: Springer Nature Switzerland. doi:10.1007/978-3-031-20059-5_1.
Lanfredi RB, Zhang M, Auffermann W, Chan J, Duong PA, Srikumar V, et al (2021). "REFLACX: Reports and eye-tracking data for localization of abnormalities in chest x-rays". PhysioNet. doi:10.13026/E0DJ-8498.
Lanfredi RB, Zhang M, Auffermann WF, Chan J, Duong PAT, Srikumar V, et al (2022). "REFLACX, a dataset of reports and eye-tracking data for localization of abnormalities in chest x-rays". Sci Data. 9(1):350. doi:10.1038/s41597-022-01441-z.
Parent Projects
MIMIC-Ext-CXR-QBA: A Structured, Tagged, and Localized Visual Question Answering Dataset with Question-Box-Answer Triplets and Scene Graphs for Chest X-ray Images was derived from:
Chest ImaGenome Dataset v1.0.0
MIMIC-CXR Database v2.1.0
MIMIC-CXR-JPG - chest radiographs with structured labels v2.1.0
Please cite them when using this project.
Share
    
Access
Access Policy:
Only credentialed users who sign the DUA can access the files.

License (for files):
PhysioNet Credentialed Health Data License 1.5.0

Data Use Agreement:
PhysioNet Credentialed Health Data Use Agreement 1.5.0

Required training:
CITI Data or Specimens Only Research

Discovery
DOI (version 1.0.0):
https://doi.org/10.13026/8qmz-da41

DOI (latest version):
https://doi.org/10.13026/6193-he91

Topics:
chest x-rays vqa localization scene graphs

Corresponding Author
Philip Müller
Technical University of Munich (TUM).
philip.j.mueller@tum.de

Versions
1.0.0
July 22, 2025
Files
Total uncompressed size: 26.1 GB.

Access the files
Download the ZIP file (25.3 GB)
Download the files using your terminal: wget -r -N -c -np --user kyagabajonah --ask-password https://physionet.org/files/mimic-ext-cxr-qba/1.0.0/
Folder Navigation: <base>
Name	Size	Modified
exports		
metadata		
stats		
LICENSE.txt(download)	2.5 KB	2025-06-09
README.md(download)	13.2 KB	2025-03-17
SHA256SUMS.txt(download)	11.7 KB	2025-07-22
qa.zip(download)	6.9 GB	2025-03-17
quality_mappings.csv(download)	4.8 KB	2025-03-17
scene_data.zip(download)	1.1 GB	2025-03-17

PhysioNet
Maintained by the MIT Laboratory for Computational Physiology

Supported by the National Institute of Biomedical Imaging and Bioengineering (NIBIB), National Heart Lung and Blood Institute (NHLBI), and NIH Office of the Director under NIH grant numbers U24EB037545 and R01EB030362

Navigation
Discover Data
Share Data
About
News
Explore
Data
Software
Challenges
Tutorials


Accessibility