import * as dotenv from "dotenv";
dotenv.config();

import { ChatOpenAI } from "@langchain/openai";
import { createSqlQueryChain } from "langchain/chains/sql_db";
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";
import { QuerySqlTool } from "langchain/tools/sql";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";

import readline from "readline";
import { startSpinner, stopSpinner } from "../utils/spinner.js";

const dbConfig = {
  type: "postgres",
  host: process.env.DB_HOST,
  port: process.env.PORT,
  username: process.env.DB_USER,
  password: process.env.DB_PASS,
  database: process.env.DB_NAME,
};

const datasource = new DataSource(dbConfig);

const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
  includesTables: ["user", "tenant", "project", "projectDataSource"],
});

const llm = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0,
});

const executeQuery = new QuerySqlTool(db);
const writeQuery = await createSqlQueryChain({
  llm,
  db,
  dialect: "postgres",
});

const answerPrompt =
  PromptTemplate.fromTemplate(`Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: `);

const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());

const chain = RunnableSequence.from([
  RunnablePassthrough.assign({ query: writeQuery }).assign({
    result: (i) => executeQuery.invoke(i.query),
  }),
  answerChain,
]);

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
  const response = await chain.invoke({ question });
  const endTime = Date.now();
  stopSpinner();

  console.log(
    `> Answer (took ${(endTime - startTime) / 1000} seconds): \n`,
    response,
  );
}
