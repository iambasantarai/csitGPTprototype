"use strict";
import dotenv from "dotenv";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain } from "langchain/chains";
import { PromptTemplate } from "@langchain/core/prompts";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";

import readline from "readline";
import { startSpinner, stopSpinner } from "../utils/spinner.js";

dotenv.config();

const loader = new DirectoryLoader("./docuemnts/", {
  ".pdf": (path) => new PDFLoader(path),
});

const documents = await loader.load();

const QA_PROMPT_TEMPLATE = `Use the following pieces of context to craft a comprehensive answer for the question at the end.
    As a helpful assistant, your answers should be concise and accurate.
    Avoid providing fabricated information if uncertain; simply acknowledge the lack of knowledge.
    {context}
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

let conversationHistory = [];

const model = new OpenAI({
  model: "gpt-3.5-turbo",
  apiKey: process.env.OPENAI_API_KEY,
});

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const normalizedDocs = normalizeDocuments(documents);
const splitDocs = await textSplitter.createDocuments(normalizedDocs);

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
});

const vectorStore = await Chroma.fromDocuments(splitDocs, embeddings, {
  collectionName: "knowledge-collection",
  url: process.env.CHROMA_URL,
});

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
  const response = await chain.call({
    query: question,
    chat_history: conversationHistory,
  });
  const endTime = Date.now();
  stopSpinner();

  conversationHistory.push([question, response.text]);

  console.log(
    `> Answer (took ${(endTime - startTime) / 1000} seconds): \n`,
    response.text,
  );
}
