"use strict";
import dotenv from "dotenv";
import fs from "fs";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAI } from "langchain/llms/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain } from "langchain/chains";
import readline from "readline";
import { startSpinner, stopSpinner } from "./utils/spinner.js";

dotenv.config();

const loader = new DirectoryLoader("./docuemnts/", {
  ".pdf": (path) => new PDFLoader(path),
});

const documents = await loader.load();

const VECTOR_STORE_PATH = "vectorStore";

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

  let vectorStore;

  if (fs.existsSync(VECTOR_STORE_PATH)) {
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
  } else {
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });

    const normalizedDocs = normalizeDocuments(documents);
    const splitDocs = await textSplitter.createDocuments(normalizedDocs);

    vectorStore = await HNSWLib.fromDocuments(
      splitDocs,
      new OpenAIEmbeddings(),
    );
    await vectorStore.save(VECTOR_STORE_PATH);
  }

  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

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
