import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const messages = [
  new SystemMessage("Translate the following from English into Italian"),
  new HumanMessage("hi!"),
];

import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({apiKey: `${process.env.OPENAI_API_KEY}`, model: "gpt-4"});

const app = async () => await model.invoke(messages);
app();