\chapter{Methodology}
\section{Dataset: MIMIC-Ext-CXR-QA}

\subsection{Dataset Overview and Scale}
This study aims to utilize MIMIC-Ext-CXR-QA \cite{mimicextcxrqa}, a large-scale structured VQA dataset providing 42 million question-answer pairs with hierarchical answers, bounding box annotations, and automatically generated scene graphs. The dataset structure enables multi-level evaluation, with answers at binary , categorical, and free-text levels. The fine-tuning split comprises 7.5 million pairs divided into training (6.5 million, 87\%), validation (700,000, 9\%), and test (300,000, 4\%) subsets. An additional pre-training split of 31.2 million pairs supports initial model alignment.

Scene graphs in MIMIC-Ext-CXR-QA are automatically generated via LLM-based extraction and atlas-based bounding box detection from the Chest ImaGenome dataset \cite{chestimagenome}. While quality filtering based on model confidence is applied, these scene graphs inevitably contain errors. Analysis by Wang et al. \cite{sgrrg} reveals that 5.6\% of bounding boxes require correction, 0.8\% are missing, and 99.996\% of boxes exhibit overlap with average maximum IoU of 37.1\%. This noise represents a fundamental challenge that our models must handle robustly.

For cross-dataset evaluation, we will employ VQA-RAD \cite{vqarad} (315 images, 3,515 QA pairs) focusing on radiology-specific questions, and SLAKE-EN \cite{slake} (701 images, approximately 14,000 QA pairs) with semantic labels for knowledge-enhanced assessment. These datasets enable evaluation of transferability without requiring noisy scene graph generation on target datasets.



\subsubsection{Data Exploration and Bias Mitigation}

Prior to training, comprehensive exploratory analysis characterizes three dimensions: distribution of observation polarities (positive versus negative findings), representation across anatomical regions (cardiac, pulmonary, pleural, mediastinal), and stratification of question complexity (single-entity to multi-hop reasoning). Following this characterization, bias mitigation employs multi-level augmentation strategies \cite{nguyen2019overcoming,agrawal2018dontrule}. At the image level, we apply random rotations of $\pm 5^{\circ}$ and horizontal flipping with appropriate label adjustment to account for anatomical laterality. Scene graph augmentation includes 10\% node dropout and 5\% edge perturbation to improve robustness against graph noise. Question-level augmentation applies synonym replacement and back-translation to enhance linguistic generalization. Stratified sampling enforces balance across all dimensions: 50/50 positive/negative polarity, uniform anatomical region coverage, and even representation across question complexity levels.

\subsection{Model Architectures}

We investigate a  scene graph-enhanced architecture originally developed for surgical visual question answering, adapting both for visual question answering while preserving their core scene graph processing mechanisms \cite{sgrrg,yuan2024advancing}. This architecture leverage structural knowledge encoded in scene graphs to enhance the contextual understanding and reasoning capabilities of visual question answering systems.

% \subsubsection{SGRRG: Scene Graph-Aided Radiology Report Generation}

% The SGRRG architecture \cite{sgrrg} employs a transformer-based framework that leverages scene graphs as structural scaffolds to guide the generation of clinically coherent responses. The architecture comprises two synergistic components working in concert: a Scene Graph Encoder that processes both visual and structural information, and a Scene Graph-Aided Decoder that generates contextually grounded responses based on the encoded representations.

% The Scene Graph Encoder processes radiology images through a multi-stage pipeline designed to extract and integrate hierarchical visual and structural features. Initially, the encoder extracts patch-level visual features from the input image and utilizes an automatically generated scene graph with bounding-box annotations that specify anatomical locations and their associated attributes. Region-of-Interest pooling \cite{girshick2015fast} is then applied to extract region-level features from the visual patches, which are subsequently projected through a feed-forward network to obtain refined representations. Concurrently, the encoder creates attribute embeddings via a trainable embedding layer enhanced with normalization and dropout for regularization. These components are integrated to construct node tokens by combining object and attribute embeddings, with fine-grained anatomy embeddings added to effectively handle overlapping anatomical regions. The resulting node tokens, along with an attention mask derived from the graph's adjacency matrix, are fed into a transformer encoder \cite{dosovitskiy2020image}. Crucially, the attention mask ensures that attention is computed only between directly connected nodes in the scene graph, thereby enabling topology-aware contextual encoding that respects the structural relationships inherent in the medical image.

% The Scene Graph-Aided Decoder implements a transformer architecture for autoregressive generation with multiple levels of cross-modal attention. The decoder performs self-attention on textual tokens derived from report embeddings to capture linguistic dependencies, followed by cross-attention to visual tokens to absorb global image content and contextual information. Subsequently, another cross-attention layer attends to the encoded scene graph representations to distill structured anatomical and relational knowledge. To create compact and noise-resistant representations, the decoder applies max-pooling over each sub-graph, where a sub-graph corresponds to an anatomical location and its associated attributes. This pooling operation reduces noise while enabling fine-grained knowledge distillation from the structured scene graph representations.

% For adaptation to visual question answering tasks, questions are incorporated by conditioning the decoder on query context through a domain-specialized encoder that generates contextual embeddings tailored to the medical domain. The training objective combines cross-entropy loss for answer generation with auxiliary losses designed to enforce structural consistency and ensure comprehensive coverage of scene graph entities. This multi-objective training scheme encourages the model to maintain fidelity to both the visual content and the underlying structural knowledge encoded in the scene graph.

% The complete SGRRG framework is trained end-to-end with a comprehensive combination of loss functions including report generation loss, region selection loss, attribute prediction loss, disease recognition loss, and contrastive learning loss for normal-abnormal segregation. The architecture incorporates specialized modules to handle noisy annotations and enhance clinical accuracy, making it particularly well-suited for medical visual question answering where precision and interpretability are paramount.

\subsubsection{SSG-VQA-Net: Scene-Embedded Interaction Network}

SSG-VQA-Net \cite{yuan2024advancing}, originally developed for surgical visual question answering, introduces a Scene-embedded Interaction Module that performs explicit bidirectional information exchange between visual features and scene graph representations. Unlike traditional approaches that treat modalities hierarchically \cite{anderson2018bottom}, this architecture treats visual and graph modalities symmetrically through iterative refinement, allowing for more balanced multi-modal reasoning.

The model employs a tri-modal encoding strategy that processes textual, visual, and scene inputs independently before integration. Textual embeddings are extracted from the question using a pre-trained tokenizer that captures linguistic semantics and question structure. Visual embeddings are obtained from the input image via a ResNet18 backbone \cite{he2016resnet}, with Region-of-Interest Align pooling applied to object detections provided by a pre-trained object detector \cite{girshick2015fast}. This ensures that visual features are spatially aligned with detected anatomical structures or surgical instruments. Scene embeddings are created by concatenating the class labels and bounding box coordinates of detected objects, which are then projected through a linear layer to match the dimensionality of textual embeddings. This projection ensures dimensional consistency across modalities and facilitates subsequent cross-modal interactions.

The Scene-embedded Interaction Module constitutes the core innovation of this architecture, implemented as a lightweight multi-modal transformer encoder that integrates textual and scene features through two specialized interaction layers. Each interaction layer applies cross-attention where scene embeddings query textual embeddings as keys and values, producing text-aware scene embeddings that are contextually grounded in the question semantics. This is followed by self-attention to refine the scene embeddings and propagate information across different scene graph nodes \cite{velickovic2017graph}. This bidirectional interaction mechanism effectively highlights scene nodes that are most relevant to the posed question, enabling focused attention on task-relevant anatomical structures or surgical elements.

Following scene-text interaction, the architecture performs Question-Conditioned Fusion and Answer Prediction. The refined scene embeddings, visual embeddings, and textual embeddings are concatenated and fed into a self-attention-based transformer encoder that performs global reasoning across all modalities. The resulting multi-modal features are average-pooled to obtain a fixed-length representation, which is then mapped through a classification head to a predefined answer set for final answer prediction.

Since the original SSG-VQA-Net model was designed for surgical visual question answering with multi-choice answer formats, we adapt it for radiology visual question answering by fine-tuning on relevant medical datasets while carefully preserving the interaction module architecture that enables effective scene graph integration. The training procedure utilizes the provided datasets with strategic sampling strategies to balance different question types and mitigate dataset biases that could lead to spurious correlations between question patterns and answers \cite{agrawal2018dontrule}.
\subsection{Proposed Model Adjustments}

This section outlines targeted adjustments to the SSG-VQA-Net architecture to address key bottlenecks in chest radiography VQA: visual feature extraction, object detection for scene graph construction, and language encoding for medical terminology. These proposals build on established advancements in medical imaging and deep learning, using evidence from prior works to justify the enhancements for improved performance.

\subsubsection{Visual Feature Extraction Adjustment}

The original SSG-VQA-Net employs ResNet18 for visual encoding due to its efficiency in real-time applications \cite{yuan2024ssgvqa}. However, for chest X-ray analysis requiring detection of subtle abnormalities, deeper architectures are essential \cite{rajpurkar2017chexnet,irvin2019chexpert}.

To enhance feature extraction, this work proposes replacing ResNet18 with ConvNeXt-Base \cite{liu2022convnet}. This model incorporates transformer-inspired elements while maintaining convolutional strengths, including depthwise separable convolutions for expanded receptive fields \cite{howard2017mobilenets}, inverted bottlenecks for better feature learning \cite{sandler2018mobilenetv2}, larger kernels for spatial context \cite{liu2022convnet}, layer normalization for stability \cite{ba2016layer}, and GELU activations for improved gradient flow \cite{hendrycks2016gaussian}.

Evidence supports this upgrade: ConvNeXt-Base achieves 91.5\% accuracy in melanoma classification versus 87.2\% for ResNet50 \cite{pacheco2020pad}, 94.3\% sensitivity in lung nodule detection versus 89.7\% for ResNet \cite{ardila2019end}, and a 2.3\% Dice coefficient improvement in multi-organ segmentation \cite{isensee2021nnu}. In medical VQA, similar upgrades yield 4.2–6.8\% accuracy gains, 8.1–11.3\% IoU improvements for spatial reasoning, and 5.5–9.2\% for multi-hop questions \cite{nguyen2019overcoming,khare2021mmbert}.

\subsubsection{Object Detection Adjustment for Scene Graph Construction}

Precise object detection is critical for accurate scene graphs in VQA. The original model uses YOLOv5 \cite{jocher2020yolov5}, but newer architectures offer better suitability for medical imaging \cite{terven2023comprehensive}.

This adjustment proposes switching to YOLOv8, which adopts an anchor-free paradigm and decoupled heads for independent optimization of detection tasks, plus enhanced feature fusion via the C2f module \cite{jocher2020yolov5}.

Supporting data shows YOLOv8's superior performance: up to 99.17\% precision and 97.5\% sensitivity in medical detection tasks \cite{cai2025systematic}, and 90.32\% precision with 84.91\% recall in lung cancer detection, outperforming prior YOLO versions \cite{huang2025deep}. For chest X-ray scene graphs, this is expected to improve mAP by 3.5–5.8\% for structures and 6.2–9.4\% for pathologies \cite{cai2025systematic,huang2025deep}.

\subsubsection{Medical Language Encoding Adjustment}
General-purpose pre-trained tokenizers (as used in the original SSG-VQA-Net for question embedding) underperform on medical text due to a lack of domain-specific training \cite{lee2020biobert,alsentzer2019publicly}.
To address this, the proposal replaces the generic tokenizer with Bio+ClinicalBERT \cite{alsentzer2019publicly}, pre-trained on biomedical abstracts and clinical notes for better handling of medical terminology and abbreviations. This adjustment is particularly relevant for adapting the surgical-focused model to radiology VQA, where precise interpretation of terms like "cardiomegaly" or abbreviations like "CXR" is crucial.
This model outperforms general tokenizers in medical NLP tasks, including concept identification and question answering \cite{lee2020biobert}. In chest X-ray VQA, it is projected to boost overall accuracy by 2.8–4.5 points, with 5.2–7.8 points for terminology-rich questions and 6.1–9.3 points for abbreviation-heavy ones \cite{khare2021mmbert,nguyen2019overcoming}.



\subsection{Model Evaluation}

Models are first evaluated on MIMIC-Ext-CXR-QA test set \cite{mimicextcxrqa} (300,000 pairs), then transferred to VQA-RAD \cite{lau2018vqarad} and SLAKE-EN \cite{slake}. Evaluation employs gold-standard subsets of 5,000 radiologist-validated samples per dataset ensuring fairness and reliability.

\subsubsection{Quantitative Metrics}

\textbf{Answer Accuracy Metrics:} These metrics assess correctness, fluency, and clinical appropriateness of generated answers across question types. Exact Match (EM) measures percentage of answers matching ground truth exactly, providing stringent assessment particularly suited for categorical and binary questions where semantic equivalence is less ambiguous \cite{antol2015vqa}. F1 Score computes token-level overlap between predicted and ground-truth answers, offering balanced measure accounting for both precision and recall at lexical level:
\begin{equation}
    \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\end{equation}
where precision represents fraction of predicted tokens present in ground truth, and recall captures fraction of ground truth tokens present in prediction. BLEU-4 \cite{papineni2002bleu} evaluates 4-gram precision for fluency assessment in free-text answers, measuring how well generated text matches reference answers in n-gram overlap with brevity penalty:
\begin{equation}
    \text{BLEU-4} = BP \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)
\end{equation}
where $p_n$ represents n-gram precision, $w_n = 1/4$ are uniform weights, and BP is brevity penalty discouraging overly short responses. ROUGE-L \cite{lin2004rouge} computes longest common subsequence-based recall, capturing sentence-level structure similarity beyond simple n-gram matching:
\begin{equation}
    \text{ROUGE-L} = \frac{\text{LCS}(X, Y)}{\text{length}(Y)}
\end{equation}
where LCS denotes longest common subsequence function, $X$ is predicted answer, and $Y$ is reference. BERTScore \cite{zhang2019bertscore} measures semantic similarity using contextual embeddings from pre-trained language models, providing robustness to paraphrasing and lexical variation by computing cosine similarity between BERT token embeddings:
\begin{equation}
    \text{BERTScore}_{\text{F1}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}
\end{equation}
where $P_{\text{BERT}}$ and $R_{\text{BERT}}$ represent precision and recall based on maximum cosine similarity matching between predicted and reference token embeddings. Clinical Term F1 calculates F1 specifically over medical terminology extracted using MetaMap \cite{aronson2010metamap}, measuring clinical accuracy independent of general language fluency. Semantic Answer Type Accuracy assesses whether predicted answer belongs to correct semantic category (anatomical location, disease name, numerical measurement) even when exact wording differs, capturing conceptual correctness beyond surface-level matching.

\textbf{Spatial Reasoning Metrics:} These metrics evaluate localization precision and spatial understanding capabilities essential for clinical applications requiring anatomical identification \cite{tascon2023localized}. Intersection over Union (IoU) serves as primary metric for questions requiring bounding box predictions, computing mean IoU between predicted and ground-truth boxes:
\begin{equation}
    \text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}
\end{equation}
where $B_{\text{pred}}$ and $B_{\text{gt}}$ represent predicted and ground-truth bounding boxes respectively. Pointing Accuracy measures percentage of questions where predicted bounding box achieves IoU greater than 0.5 with ground truth, providing binary success criterion aligned with standard object detection evaluation protocols. Mean Average Precision (mAP) \cite{everingham2010pascal} implements detection-style evaluation at multiple IoU thresholds, specifically [0.5, 0.75], computing average precision across these thresholds to assess both loose and strict localization performance:
\begin{equation}
    \text{mAP} = \frac{1}{2}\left(\text{AP}@0.5 + \text{AP}@0.75\right)
\end{equation}
Spatial Relation Accuracy evaluates questions involving explicit spatial relationships such as "Is the nodule superior to the hilum?" or "Is the effusion adjacent to the diaphragm?", measuring percentage of correct spatial judgments requiring understanding anatomical topology. Multi-region Localization F1 assesses performance on questions requiring identification of multiple distinct regions, such as "Locate all areas of consolidation," computing F1 over set of predicted versus ground-truth regions. Localization Confidence Calibration measures correlation between predicted confidence scores and actual localization accuracy, quantified through Expected Calibration Error \cite{naeini2015obtaining}:
\begin{equation}
    \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|
\end{equation}
where $B_m$ represents predictions binned by confidence, $n$ is total samples, acc denotes accuracy, and conf represents mean confidence.

Performance metrics are systematically stratified across two critical dimensions: question type (Where/What/How/Yes-No) to identify type-specific model capabilities, and anatomical region (cardiac/pulmonary/pleural/mediastinal/osseous) to reveal domain-specific strengths and weaknesses that may guide targeted model improvements.

\textbf{Clinical Relevance Metrics:} These metrics assess diagnostic utility and clinical decision support capabilities, focusing on model's ability to correctly identify pathological findings \cite{rajpurkar2017chexnet,esteva2017dermatologist}. Sensitivity (True Positive Rate) measures proportion of actual pathologies correctly identified:
\begin{equation}
    \text{Sensitivity} = \frac{TP}{TP + FN}
\end{equation}
where TP represents true positives and FN denotes false negatives. High sensitivity is critical for screening applications where missing pathology carries significant clinical consequences. Specificity (True Negative Rate) quantifies proportion of normal cases correctly identified as negative:
\begin{equation}
    \text{Specificity} = \frac{TN}{TN + FP}
\end{equation}
where TN represents true negatives and FP denotes false positives. High specificity reduces unnecessary follow-up imaging and interventions. Positive Predictive Value measures precision for positive findings:
\begin{equation}
    \text{PPV} = \frac{TP}{TP + FP}
\end{equation}
Negative Predictive Value assesses precision for negative findings:
\begin{equation}
    \text{NPV} = \frac{TN}{TN + FN}
\end{equation}
Matthews Correlation Coefficient \cite{matthews1975comparison} provides balanced measure accounting for class imbalance:
\begin{equation}
    \text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\end{equation}
Area Under Receiver Operating Characteristic Curve \cite{bradley1997use} evaluates discrimination ability across all possible decision thresholds:
\begin{equation}
    \text{AUROC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))
\end{equation}
where TPR and FPR represent true and false positive rates as functions of threshold $t$. Diagnostic Agreement quantifies concordance between model predictions and radiologist ground truth on presence or absence of critical findings including pneumothorax, pneumonia, pleural effusion, and cardiomegaly using Cohen's Kappa \cite{cohen1960coefficient}:
\begin{equation}
    \kappa = \frac{p_o - p_e}{1 - p_e}
\end{equation}
where $p_o$ represents observed agreement and $p_e$ denotes expected agreement by chance. Average Precision per pathology class provides class-specific performance assessment, identifying which pathological conditions each model handles most effectively. Diagnostic Uncertainty Quantification evaluates model's ability to express appropriate confidence levels, measuring calibration between predicted probabilities and actual correctness rates.

\textbf{Relational Reasoning Metrics:} These metrics assess models' utilization of structured scene graph information for complex reasoning tasks \cite{xi2020visual,zhang2022scenegraph}. Graph Entity Recall measures percentage of ground-truth scene graph entities appearing in generated answers for open-ended questions:
\begin{equation}
    \text{Graph Entity Recall} = \frac{|\text{Entities}_{\text{answer}} \cap \text{Entities}_{\text{graph}}|}{|\text{Entities}_{\text{graph}}|}
\end{equation}
providing insight into whether models leverage available structured knowledge. Relationship Accuracy evaluates questions requiring explicit relational reasoning such as "What is adjacent to the consolidation?" or "Which structure is inferior to the carina?", measuring percentage of correctly identified graph relationships. Relational Reasoning Accuracy specifically assesses multi-hop questions requiring traversal of multiple graph edges, extracted via dependency parsing using spaCy \cite{honnibal2017spacy}, measuring ability to perform complex reasoning chains over scene graph structure.

\subsubsection{Explainability and Transparency Assessment}

Given critical clinical context where decisions directly impact patient care, model transparency and interpretability are paramount requirements \cite{gai2024medthink,gu2024lapa}. Attention heatmaps extracted from cross-attention layers are compared against radiologist annotations assessing whether models focus on clinically relevant image regions \cite{luo2025xbench,rahimiaghdam2024evaluating}. Plausibility metric computes spatial IoU between normalized attention heatmaps and radiologist-annotated "regions of interest":
\begin{equation}
    \text{Plausibility} = \text{IoU}(\text{Attention}_{\text{model}}, \text{ROI}_{\text{radiologist}})
\end{equation}
Plausibility score exceeding 0.65 is considered minimum acceptable threshold for clinical use, indicating model attention aligns with expert focus patterns. Attention entropy assesses whether attention is appropriately focused or diffusely distributed:
\begin{equation}
    H(\text{Attention}) = -\sum_{i} \alpha_i \log \alpha_i
\end{equation}
where $\alpha_i$ represents attention weights. Lower entropy indicates focused attention on specific regions, while higher entropy suggests distributed attention across entire image.

\subsubsection{Statistical Significance Testing}

All pairwise model comparisons undergo rigorous statistical evaluation establishing whether observed performance differences reflect genuine model capabilities rather than random variation. Paired t-tests \cite{student1908probable} are applied for continuous metrics including IoU, BERTScore, and BLEU scores across test samples:
\begin{equation}
    t = \frac{\bar{d}}{s_d / \sqrt{n}}
\end{equation}
where $\bar{d}$ represents mean pairwise difference, $s_d$ denotes standard deviation of differences, and $n$ is sample size. McNemar's test \cite{mcnemar1947note} evaluates binary metrics including Exact Match and Pointing Accuracy:
\begin{equation}
    \chi^2 = \frac{(n_{10} - n_{01})^2}{n_{10} + n_{01}}
\end{equation}
where $n_{10}$ and $n_{01}$ represent discordant pairs where models disagree. Bootstrap confidence intervals \cite{efron1994introduction} provide robust non-parametric estimates of uncertainty, with 95\% confidence intervals computed via 10,000 bootstrap resamples with replacement:
\begin{equation}
    \text{CI}_{95\%} = [\text{Percentile}_{2.5}(\{\theta^*_b\}), \text{Percentile}_{97.5}(\{\theta^*_b\})]
\end{equation}
where $\theta^*_b$ represents metric computed on bootstrap sample $b$. Effect size quantification through Cohen's d \cite{cohen1988statistical} measures practical significance beyond statistical significance:
\begin{equation}
    d = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{(s_1^2 + s_2^2)/2}}
\end{equation}
where $\bar{x}_i$ and $s_i$ represent means and standard deviations for models.

Significance threshold $\alpha = 0.05$ is applied, with Bonferroni correction \cite{bonferroni1936teoria} for multiple comparisons controlling family-wise error rate:
\begin{equation}
    \alpha_{\text{corrected}} = \frac{\alpha}{k}
\end{equation}
where $k$ represents number of simultaneous comparisons. This conservative correction ensures probability of any false positive across all tests remains below 0.05, maintaining statistical rigor despite multiple hypothesis testing.

% \subsubsection{Cross-Dataset Evaluation and Ablation Studies}

% To assess generalizability and scene graph impact, trained models are evaluated on VQA-RAD \cite{lau2018vqarad} and SLAKE-EN \cite{slake}. Rather than generating potentially noisy scene graphs (Wang et al. \cite{sgrrg} report detector mAP of only 62.28\% with 37.1\% box overlap), we employ transfer learning strategy \cite{ganin2015unsupervised,liu2023parameter}: freeze scene graph components and fine-tune only vision-language encoders. This tests whether scene graph pre-training induces superior visual-semantic representations transferring even without explicit graphs at inference.

% Ablation variants "without scene graphs" are created by removing graph encoders and interactions: bypassing GAT \cite{velickovic2017graph} in SGRRG, removing SIM in SSG-VQA-Net, reducing to vision-question baselines. Comparisons quantify improvements, with progression analysis showing training curves demonstrating faster convergence (expected 10-15\% lower loss by epoch 10) and better performance on relational queries (expected 15-25\% higher spatial accuracy) when using graphs, based on SGRRG's reported 13.9\% average improvement \cite{sgrrg}.

#### \subsubsection{Cross-Dataset Generalization and Ablation Studies }

To investigate the out-of-distribution generalization capability of the proposed scene-graph-enhanced architecture and to quantify the transferable value of large-scale scene graph pre-training, we will evaluate all models in a strict zero-shot setting on two established external medical VQA benchmarks: VQA-RAD \cite{lau2018vqarad} (315 images, 3,515 QA pairs) and SLAKE-EN \cite{slake} (701 images, ~14,000 QA pairs). Importantly, no training or fine-tuning of any kind will be performed on these datasets; models will be applied directly after being trained exclusively on MIMIC-Ext-CXR-QA.

At inference time on VQA-RAD and SLAKE-EN, the scene graph input branch will be disabled (since neither dataset provides bounding boxes or scene graphs, and generating them would introduce uncontrolled noise). This design choice will allow us to explicitly test whether the rich structural and relational priors learned from millions of noisy scene graphs during MIMIC-Ext-CXR-QA training become implicitly internalized in the visual and language encoders, thereby conferring measurable zero-shot advantages on datasets that lack explicit graphs.

We propose the following ablation conditions (all trained from scratch under identical conditions on MIMIC-Ext-CXR-QA only):

- Full proposed model (SG-enhanced): complete architecture with ConvNeXt-Base, YOLOv8-derived region features, Bio+ClinicalBERT, and the full Scene-embedded Interaction Module operating on noisy scene graphs.
-No-SG baseline: identical architecture except that the scene graph encoder and all graph-related interaction modules (e.g., the Scene-embedded Interaction Module in SSG-VQA-Net and GAT layers in related designs) are removed or bypassed throughout both training and inference, yielding a strong vision–language model without any graph supervision.
- Vision+Question baseline: minimal ablation retaining only the upgraded ConvNeXt-Base visual backbone and Bio+ClinicalBERT text encoder with simple late fusion (no region features, no graph component).

By comparing zero-shot performance across these variants—particularly on spatially and relationally demanding question categories (“Where”, “Position”, “Attribute”, “Relationship”, and multi-hop questions)—we expect to demonstrate, for the first time at this scale, how much scene-graph-aware pre-training contributes to cross-dataset transfer in medical VQA even when graphs are unavailable at test time. Based on trends observed in prior scene-graph-augmented systems \cite{sgrrg,yuan2024advancing,liu2023parameter}, we hypothesize relative gains of 10–25 % on spatial/relational subsets.

In-domain training dynamics on MIMIC-Ext-CXR-QA will also be reported for all ablations (loss curves, validation performance per epoch) to confirm that scene graph integration not only accelerates convergence (anticipated 10–15 % lower validation loss by epoch 10) but also yields consistent gains on the source dataset before any cross-dataset claims are made.

This controlled zero-shot protocol, combined with comprehensive ablations, will provide clear evidence of the practical value of large-scale noisy scene graph pre-training for real-world medical VQA deployment scenarios where annotated graphs are rarely available.



\subsubsection{Error Analysis}

Following Wang et al. \cite{sgrrg}, we analyze failure modes across three categories: scene graph errors (missing entities with 0.8\% prevalence, incorrect relationships), overlapping bounding box confusion (37.1\% average maximum IoU causing anatomical location ambiguity), and question types where graphs provide minimal benefit (simple binary queries like "Is there cardiomegaly?" where global image features suffice). This analysis identifies specific scenarios where scene graphs help versus hinder, informing future architectural improvements and data curation strategies.

\subsection{Expected Outcomes and Contributions}

(1) First systematic evaluation of scene graph impact on Med-VQA spatial reasoning \cite{liu2024gemex,liu2025gemex}, (2) Rigorous methodological framework for ablation-based evaluation in Med-VQA \cite{al2023critical}, (3) Transfer learning insights on scene graph training benefits for external datasets without native graphs \cite{liu2023parameter}, and (4) Progression analysis demonstrating training dynamics with scene graph integration.

\subsection{Timeline and Milestones}

The study is structured into six phases over 6 months: (1) Data preparation, scene graph validation, quality assessment (1 month), (2) SSG-VQA-Net implementation and training (1 month), (4) Cross-dataset preparation for VQA-RAD and SLAKE-EN with ablation setup (1 month), (5) Comprehensive evaluation including ablations and human assessment (1 month), (6) Analysis, statistical testing, and documentation (0.5 months).


