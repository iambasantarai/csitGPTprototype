"use strict";
import dotenv from "dotenv";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";
import { Chroma } from "@langchain/community/vectorstores/chroma";

import readline from "readline";
import { startSpinner, stopSpinner } from "./utils/spinner.js";

dotenv.config();

const loader = new DirectoryLoader("./docuemnts/", {
  ".pdf": (path) => new PDFLoader(path),
});

const documents = await loader.load();

const QA_PROMPT_TEMPLATE = `Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    You are a helpful assistant in completing questions using given piceces of context before answering.
    Question: {question}
    Helpful Answer:`;

function normalizeDocuments(documents) {
  return documents.map((document) => {
    if (typeof document.pageContent === "string") {
      return document.pageContent;
    } else if (Array.isArray(document.pageContent)) {
      return document.pageContent.join("\n");
    }
  });
}

(async () => {
  const model = new OpenAI({
    temperature: 0,
    modelName: "gpt-3.5-turbo",
  });

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const normalizedDocs = normalizeDocuments(documents);
  const splitDocs = await textSplitter.createDocuments(normalizedDocs);

  const vectorStore = await Chroma.fromDocuments(
    splitDocs,
    new OpenAIEmbeddings(),
    {
      collectionName: "default-collection",
      url: process.env.CHROMA_URL,
    },
  );

  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
    prompt: PromptTemplate.fromTemplate(QA_PROMPT_TEMPLATE),
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const question = await new Promise((resolve) => {
      rl.question("> Question: ", resolve);
    });

    if (question.toLowerCase() == "quit") {
      rl.close();
      break;
    }

    startSpinner();
    const startTime = Date.now();
    const response = await chain.call({ query: question });
    const endTime = Date.now();
    stopSpinner();

    console.log(
      `> Answer (took ${(endTime - startTime) / 1000} seconds): \n`,
      response.text,
    );
  }
})();
