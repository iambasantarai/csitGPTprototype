import * as dotenv from "dotenv";
dotenv.config();

import readline from "readline";

import { ChatOpenAI } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { startSpinner, stopSpinner } from "../utils/spinner.js";

import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { Calculator } from "@langchain/community/tools/calculator";
import { GoogleCustomSearch } from "@langchain/community/tools/google_custom_search";

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
  temperature: 0.2,
});

const prompt = ChatPromptTemplate.fromMessages([
  ("system", "You are a helpful study assistant named TURTY."),
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

const tools = [travilySearchTool, calculatingTool, googleSearchTool];

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
