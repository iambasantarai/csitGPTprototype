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

dotenv.config();

const loader = new DirectoryLoader("./docuemnts/", {
  ".pdf": (path) => new PDFLoader(path),
});

console.log("Loading documents...");
const documents = await loader.load();
console.log("Documents loaded.");

const VECTOR_STORE_PATH = "vectorStore";
const question = "How to use GHC(i) in haskell ?";

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

  console.log("Checking for existing vector store...");
  if (fs.existsSync(VECTOR_STORE_PATH)) {
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
    console.log("Vector store loaded.");
  } else {
    console.log("Creating new vector store...");

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

    console.log("Vector store created.");
  }
  console.log("Creating retrieval chain...");
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

  console.log("Querying chain...");
  const startTime = Date.now();
  const response = await chain.call({ query: question });
  const endTime = Date.now();

  console.log(`ANSWER (took ${(endTime - startTime) / 1000}sec) : `, response);
})();
