<p align="center">
    <img alt="csitGPT" src="./assets/csitGPT.png">
</p>

---

csitGPT is a command line interface (CLI) application tailored to provide accurate and insightful responses to `CSIT` related queries.
Through the integrating cutting-edge technologies such as [LangChain](https://www.langchain.com/), Large Language Models (LLMs) ([Hugging Face](https://huggingface.co/)/[OpenAI](https://openai.com/)), and [ChromaDB](https://www.trychroma.com/), csitGPT delivers a powerful and insightful Q&A chatbot experience.

### Getting started

1. Clone the repository

```bash
git clone git@github.com:iambasantarai/csitGPTprototype.git
```

2. Navigate to the project directory

```bash
cd csitGPTprototype
```

3. Install dependencies

```bash
yarn install
```

OR

```bash
yarn install
```

4. Configure environment

```bash
cp .env.example .env
```

Update the values for following keys in the `.env`

- `CHROMA_URL`
- `HUGGINGFACEHUB_API_KEY`
- `OPENAI_API_KEY`

5. Run the script

```bash
yarn run bot
```
