import * as dotenv from "dotenv";
dotenv.config();

import readline from "readline";

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { startSpinner, stopSpinner } from "../utils/spinner.js";

import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Chroma } from "@langchain/community/vectorstores/chroma";

import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { Calculator } from "@langchain/community/tools/calculator";
import { GoogleCustomSearch } from "@langchain/community/tools/google_custom_search";
import { createRetrieverTool } from "langchain/tools/retriever";

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
  temperature: 0.2,
});

const prompt = ChatPromptTemplate.fromMessages([
  ("system",
  "You are a helpful study assistant named TURTY." +
    "Your answers sholuld be concise and accurate to avoid any confusion." +
    " Avoid providing fabricated information if uncertaion; simply acknowledge the lack of knowledge."),
  new MessagesPlaceholder("chat_history"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
]);

// Define the tools the agent will have access to.
const travilySearchTool = new TavilySearchResults();

const calculatingTool = new Calculator({
  name: "calculating_tool",
  description: "useful for when you need to answer questions about math",
});

const googleSearchTool = new GoogleCustomSearch({
  name: "google_search",
  description: "Search Google for recent results.",
  apiKey: process.env.GOOGLE_API_KEY,
  googleCSEId: process.env.GOOGLE_CSE_ID,
});

const loader = new DirectoryLoader("./docuemnts/", {
  ".pdf": (path) => new PDFLoader(path),
});

const documents = await loader.load();

function normalizeDocuments(documents) {
  return documents.map((document) => {
    if (typeof document.pageContent === "string") {
      return document.pageContent;
    } else if (Array.isArray(document.pageContent)) {
      return document.pageContent.join("\n");
    }
  });
}

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

const retrieverTool = createRetrieverTool(vectorStore.asRetriever(), {
  name: "retriever_agent",
  description:
    "Use this tool when searching for information about the course details of Bachelors of Computer Science and Information Technology (BSc.CSIT)",
});

const tools = [
  travilySearchTool,
  calculatingTool,
  googleSearchTool,
  retrieverTool,
];

const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt,
  tools,
});

const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const conversationHistory = [];

while (true) {
  const query = await new Promise((resolve) => {
    rl.question("> Query: ", resolve);
  });

  if (query.toLowerCase() == "quit") {
    rl.close();
    break;
  }

  startSpinner();
  const startTime = Date.now();

  const response = await agentExecutor.invoke({
    input: query,
    chat_history: conversationHistory,
  });

  const endTime = Date.now();
  stopSpinner();

  conversationHistory.push(new HumanMessage(query));
  conversationHistory.push(new AIMessage(response.output));

  console.log(
    `> Answer (generated in ${(endTime - startTime) / 1000} seconds): \n`,
    response.output,
  );
}
