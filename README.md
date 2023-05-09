# Large Language Models (LLMs) for Classical Languages

We will use this repo to organize our summer research program.
The goal of this project is to get LLMs working with classical languages like Latin.

## Schedule

We will generally meet 3x/week as a large team on MWF from 9AM-11AM.
For the first week, we will also meet on Tuesday and may go beyond 11AM.
At the meetings, we will:
1. review research papers together
1. assign tasks to students (individual + groups)
1. review progress on those tasks

We will use this repo for all of our research communication,
and so you should all watch the repo and post questions/comments to the issues.

**Week starting May 22:**

To prepare for our first meeting:

1. There's two papers you should read:
    1. The LLAMA paper: <https://arxiv.org/abs/2302.13971>
    1. [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712)

    You probably won't understand the vast majority of these papers,
    and that's why we'll be going over them in detail in our meetings.
    Try to understand as much of the papers as you can,
    and come up with detailed, specific questions about parts that you don't understand.
    I recommend spending about 10 hours total between the two papers.
    We'll start with the LLAMA paper, so if that's the one to prioritize in your reading.

1. The following links might help give you some context of the papers:
    1. https://magazine.sebastianraschka.com/p/understanding-large-language-models
    1. https://www.reddit.com/r/MachineLearning/comments/133styi/p_understanding_large_language_models_a/
    1. https://agi-sphere.com/llama-models/

Our goal for the week will be to:

1. Understand the papers above and how LLMs work and how they fit into the Machine Learning/Natural Language Processing ecosystem.

1. Assign everyone tasks to work on and get started on those tasks.
    The tasks vary tremendously in difficulty, and so some can be completed in just a few days and some may not be completed over the course of the summer.

**Things we'll read in the future:**

About Latin:
1. About LLPSI and other resources: https://scholaeinterretiales.wordpress.com/teach-yourself-latin/
1. LLPSI audiobook: https://www.youtube.com/watch?v=t_Hm6HpnN5k&list=PLU1WuLg45SiyrXahjvFahDuA060P487pV&index=4

BLUE score and machine translation:
1. https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b

Finetuning:
1. [LlamaAcademy: Teach GPTs to understand API documentation with LoRA](https://news.ycombinator.com/item?id=35634120)
1. [DyLoRA: Parameter Efficient Tuning of Pre-Trained Models](https://news.ycombinator.com/item?id=35514228)

## Tasks

1. Create a table of LLMs and languages that they were trained on.

1. Create a table of how many GPT/BERT tokens are used by each language on the same data.

   Use the bible dataset to support a large range of languages:
   1. https://opus.nlpl.eu/bible-uedin.php
   1. https://aclanthology.org/2021.computel-1.6.pdf
   1. Also important to use various English translations/paraphrases and not just one

1. Prompt design for the existing LLPSI data.
    1. How does the choice of model affect prompt?  We can run LLAMA/etc locally on the Lambda server, or 
    1. Standards for prompt design?

1. Finetune new LLMs
    1. This will be easiest to do using the OpenAI finetuning API, but we could also do it a bit on the lambda server/other cloud platforms.

1. Train non-LLM models (RNN/Bert/etc) from scratch.

<!--
1. Load more data.
    1. More LLPSI.
    1. Other sources:

    1. Latin:
        1. Pre-training data:
            1. Latin Wikipedia
            1. Latin 
        1. Textbooks:
            1. LLPSI
            1. Latin by the Natural Method
    1. Other languages:
        1. Greek
        1. Quechua
        1. Nahuatl 
-->

