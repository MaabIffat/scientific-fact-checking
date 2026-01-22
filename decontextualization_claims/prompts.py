# decontextualization_claims/prompts.py

EXAMPLES_BLOCK_TASK1 = """
Example 1:
CONTEXT: Title: Improving Arabic-to-English Statistical Machine Translation by Reordering Post-Verbal
Subjects for Alignment.
Abstract: “We study the challenges raised by Arabic verb and subject detection and reordering in
Statistical Machine Translation (SMT). We show that post-verbal subject (VS) constructions are hard to
translate because they have highly ambiguous reordering patterns when translated to English. In addition
implementing reordering is difficult because the boundaries of VS constructions are hard to detect
accurately, even with a state-of-the-art Arabic dependency parser. We therefore propose to reorder VS
constructions into SV order for SMT word alignment only. This strategy significantly improves BLEU and
TER scores, even on a strong large-scale baseline and despite noisy parses.”
CLAIM: Standard phrase-based SMT systems do not capture any generalizations between occurrences in VS
and SV orders. Label: Support
DECONTEXTUALIZED_CLAIM: Standard phrase-based Statistical Machine Translation (SMT) systems do not
capture any generalizations between occurrences of Arabic post-verbal subject (VS) constructions and
subject-verb (SV) orders.

Example 2:
CONTEXT: Title: Sense-Aware Neural Models for Pun Location in Texts.
Abstract: “A homographic pun is a form of wordplay in which one signifier (usually a word) suggests two
or more meanings by exploiting polysemy for an intended humorous or rhetorical effect. In this paper,
we focus on the task of pun location, which aims to identify the pun word in a given short text. We
propose a sense-aware neural model to address this challenging task. Our model first obtains several WSD
results for the text, and then leverages a bidirectional LSTM network to model each sequence of word
senses. The outputs at each time step for different LSTM networks are then concatenated for prediction.
Evaluation results on the benchmark SemEval 2017 dataset demonstrate the efficacy of proposed model.”
CLAIM: The task of identifying the pun word is known as pun location, which is an easy task. Label:
Refute
DECONTEXTUALIZED_CLAIM: The task of identifying the pun word in short texts, known as pun location, is
an easy task, even when using sense-aware neural models such as a bidirectional LSTM network.

Example 3:
CONTEXT: Title: Automatic Extraction of Commonsense LOCATEDNEAR Knowledge
Abstract: “LOCATEDNEAR relation is a kind of commonsense knowledge describing two physical objects that
are typically found near each other in real life. In this paper, we study how to automatically extract
such relationship through a sentence-level relation classifier and aggregating the scores of entity
pairs from a large corpus. Also, we release two benchmark datasets for evaluation and future research.”
CLAIM: The paper proposes a method to automatically extract the commonsense LOCATEDNEAR relation between
physical objects from textual corpora. Label: Support
DECONTEXTUALIZED_CLAIM: The paper proposes a sentence-level relation classification method to
automatically extract the commonsense LOCATEDNEAR relation between physical objects from large textual
corpora.

Example 4:
CONTEXT: Title: PhraseCTM: Correlated Topic Modeling on Phrases within Markov Random Fields
Abstract: “Recent emerged phrase-level topic models are able to provide topics of phrases, which are
easy to read for humans. But these models are lack of the ability to capture the correlation structure
among the discovered numerous topics. We propose a novel topic model PhraseCTM and a two-stage method to
find out the correlated topics at phrase level. In the first stage, we train PhraseCTM, which models the
generation of words and phrases simultaneously by linking the phrases and component words within Markov
Random Fields when they are semantically coherent. In the second stage, we generate the correlation of
topics from PhraseCTM. We evaluate our method by a quantitative experiment and a human study, showing
the correlated topic modeling on phrases is a good and practical way to interpret the underlying themes
of a corpus.”
CLAIM: The correlated topic modeling on phrases is not a practical way to interpret the underlying
themes of a corpus. Label: Refute
DECONTEXTUALIZED_CLAIM: Correlated topic modeling on phrases using the PhraseCTM model is not a
practical way to interpret the underlying themes of a text corpus.

Example 5:
CONTEXT: Title: Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering
Abstract: Recent studies on semantic frame induction show that relatively high performance has been
achieved by using clustering-based methods with contextualized word embeddings. However, there are two
potential drawbacks to these methods: one is that they focus too much on the superficial information of
the frame-evoking verb and the other is that they tend to divide the instances of the same verb into too
many different frame clusters. To overcome these drawbacks, we propose a semantic frame induction method
using masked word embeddings and two-step clustering. Through experiments on the English FrameNet data,
we demonstrate that using the masked word embeddings is effective for avoiding too much reliance on the
surface information of frame-evoking verbs and that two-step clustering can improve the number of
resulting frame clusters for the instances of the same verb.”
CLAIM: The proposed method uses masked word embeddings of frame-evoking verbs in addition to standard
contextualized word embeddings of frame-evoking verbs. Label: Support
DECONTEXTUALIZED_CLAIM: The proposed semantic frame induction method using masked word embeddings and
two-step clustering employs masked word embeddings of frame-evoking verbs in addition to standard
contextualized word embeddings of frame-evoking verbs.

Example 6:
CONTEXT: Title: UltraSparseBERT: 99% Conditionally Sparse Language Modelling.
Abstract: “Language models only really need to use a tiny fraction of their neurons for individual inferences. We present UltraSparseBERT, a BERT variant that uses 0.3% of its neurons during inference
while performing on par with similar BERT models. UltraSparseBERT selectively engages just 12 out of
4095 neurons for each layer inference. This is achieved by reorganizing feedforward networks into fast
feedforward networks (FFFs). To showcase but one benefit of high sparsity, we provide an Intel MKL
implementation achieving 78x speedup over the optimized feedforward baseline on CPUs, and an OpenAI
Triton implementation performing forward passes 4.1x faster than the corresponding native GPU
implementation. The training and benchmarking code is enclosed.”
CLAIM: UltraSparseBERT uses the same number of neurons as other BERT models during inference. Label:
Refute
DECONTEXTUALIZED_CLAIM: UltraSparseBERT, a variant of BERT that employs fast feedforward networks for
conditional sparsity, uses the same number of neurons as standard BERT models during inference.
""".strip()


EXAMPLES_BLOCK_TASK2 = """
Example 1:
##CLAIM##: The researchers propose a new paradigm of grounding comparative adjectives describing colors as directions in RGB space such that the colors along the vector, rooted at the reference color, satisfy the comparison.
CONTEXT: Title: Lighter Can Still Be Dark: Modeling Comparative Color Descriptions
Abstract: We propose a novel paradigm of grounding comparative adjectives within the realm of color descriptions.
Given a reference RGB color and a comparative term (e.g., ‘lighter’, ‘darker’), our model learns to ground
the comparative as a direction in the RGB space such that the colors along the vector, rooted at
the reference color, satisfy the comparison. Our model generates grounded representations of comparative adjectives with an
average accuracy of 0.65 cosine similarity to the desired direction of change. These vectors approach colors with Delta-E
scores of under 7 compared to the target colors, indicating the differences are very small with respect to human
perception. Our approach makes use of a newly created dataset for this task derived from existing labeled color data.
Keywords: {“comparative adjectives”, “color descriptions”, “RGB color space”, “model”, “grounding”, “vector”, “cosine similarity”, “Delta-E scores”, “dataset”, “labeled color data”, “reference color”, “target color”, “direction of change”, “deep learning model”, “network architecture”, “training”, “testing”, “accuracy”, “performance”}
DECONTEXTUALIZED_CLAIM: The paper “Lighter Can Still Be Dark: Modeling Comparative Color Descriptions” proposes a novel paradigm for grounding comparative adjectives describing colors as directions in RGB space, such that the colors along the vector, rooted at the reference color, satisfy the comparison.
SUBJECT: grounding comparative adjectives within the realm of color descriptions.
DISAMBIGUATION_CRITERIA: Technique or Approach

Example 3:
##CLAIM##: The paper claims that the use of either the connective or syntactic features alone results in better disambiguation performance than using both together. Label: Refute
CONTEXT: Title: Using Syntax to Disambiguate Explicit Discourse Connectives in Text
Abstract: Discourse connectives are words or phrases such as once, since, and on the contrary that explicitly signal
the presence of a discourse relation. There are two types of ambiguity that need to be resolved during
discourse processing. First, a word can be ambiguous between discourse or non-discourse usage. For example,
once can be either a temporal discourse connective or a simply a word meaning “formerly”. Secondly, some
connectives are ambiguous in terms of the relation they mark. For example since can serve as either a
temporal or causal connective. We demonstrate that syntactic features improve performance in both disambiguation
tasks. We report state-of-the-art results for identifying discourse vs. non-discourse usage and human-level
performance on sense disambiguation.
Keywords: discourse connectives, syntactic features, discourse vs. non-discourse, ambiguity, relation sense, Penn Discourse Treebank (PDTB), Expansion, Comparison, Contingency, Temporal, maximum entropy classifier, disambiguation, implicit relations, explicit relations, NSF grants.
DECONTEXTUALIZED_CLAIM: The paper “Using Syntax to Disambiguate Explicit Discourse Connectives in Text” claims that using either connective features or syntactic features alone leads to better disambiguation performance than using both feature types together.
SUBJECT: Impact of connective and syntactic features, used alone or in combination, on disambiguation of explicit discourse connectives.
DISAMBIGUATION_CRITERIA: Feature Integration and Experimental Context.
""".strip()


def construct_prompt_task1(claim: str, title: str, abstract: str, label: str) -> str:
    return f"""
TASK 1:
DECONTEXTUALIZATION CRITERIA: Decontextualization adds the right type of information to a CLAIM to make it standalone.
This process can modify the original CLAIM in the following manners:
- Substituting pronouns or incomplete names with the specific subject being referred to.
- Including contextual information to provide more context about the subject.

Instructions:
- Identify the "subject" of the claim and locate the claim within the context.
- Use the CONTEXT to substitute any incomplete technique, the research paper, the proposed method, dataset, approach in the CLAIM.
- If there is no decontextualization necessary, return the original claim and evidence as is.
- The decontextualization should minimally modify the claim and evidence by only adding necessary contextual information.
- If the label of the CLAIM is Refute, write in the same sense rather than correcting it but complete the contextual information.

{EXAMPLES_BLOCK_TASK1}

Similarly, generate a decontextualized claim for the following pair of CLAIM and CONTEXT making minimal alterations to the original structure of the CLAIM while ensuring clarity and coherence.

CONTEXT: Title: {title}
Abstract: {abstract}
CLAIM: {claim} Label: {label}

Return ONLY the decontextualized claim as output (no extra explanation).
""".strip()


def construct_prompt_task2(decontext_claim: str, claim: str, label: str, title: str, abstract: str, keywords: str) -> str:
    return f"""
TASK 2:
AMBIGUITY CRITERIA: Ambiguity manifests in diverse forms, including:
- Similar names denoting distinct entities.
- Varied interpretations stemming from insufficient information.
- Multiple understandings arising from vague or unclear information.

Instructions:
- Identify the main SUBJECT within the claim based on the CONTEXT and KEYWORDS provided.
- Determine if the SUBJECT is ambiguous according to the provided AMBIGUITY CRITERIA.
- Utilize your world knowledge and keywords provided to enumerate potential DISAMBIGUATIONS for the identified SUBJECT.
- Specify the TYPE of information employed for disambiguation based on the list of DISAMBIGUATIONS.
- If the SUBJECT does not have ambiguous interpretations, return None

Similarly generate the subject and disambiguation criteria for the following CLAIM, CONTEXT, KEYWORDS, and DECONTEXTUALIZED_CLAIM provided in the exact same format as examples.

{EXAMPLES_BLOCK_TASK2}

CLAIM: {claim} Label: {label}
CONTEXT: Title: {title}
Abstract: {abstract}
KEYWORDS: {keywords}
DECONTEXTUALIZED_CLAIM: {decontext_claim}

Return ONLY the SUBJECT and DISAMBIGUATION_CRITERIA as output (no extra explanation).
Format:
SUBJECT: <subject>
DISAMBIGUATION_CRITERIA: <criteria>
""".strip()
