/**
 * DietPrompt engine module
 * Includes the core DietPrompt class and helper functions
 */

import {
  OpenAIApi,
  type Configuration,
  type CreateCompletionRequest,
} from "openai";
import { stopwords } from "./data";
import { Tiktoken, encoding_for_model, TiktokenModel } from "@dqbd/tiktoken";

export type CompressionType = "none" | "default";

/**
 * Handles compression of prompts and monitoring of token usage.
 *
 * This is for OpenAi's completion endpoint.
 *
 * @param {Configuration} openAiConfiguration - OpenAI configuration object
 * @param {CreateCompletionRequest} openAICompletionRequest - OpenAI completion request object
 * @param {CompressionType} compressionType - Compression type to use.
 * It's either "none" or "default". Defaults to "default". The default
 * will perform a basic compression of the prompt by removing
 * punctuation and stopwords.
 *
 * @example
 * const configuration = new Configuration({
 *  apiKey: process.env.OPENAI_API_KEY,
 * });
 *
 * const request = {
 *  model: "gpt-3.5-turbo",
 *  prompt: "Hey, this is an example request to test out DietPrompt.",
 * }
 *
 * const dietPrompt = new DietPromptCompletion(configuration, request);
 * console.log(dietPrompt.originalPrompt.tokensCount());
 * // 13
 * console.log(dietPrompt.dietPrompt.tokensCount());
 * // 6
 */
export class DietPromptCompletion {
  openAiConfiguration: Configuration;
  openAICompletionRequest: CreateCompletionRequest;
  compressionType: CompressionType;

  originalPrompt: BasePrompt;
  dietPrompt: BasePrompt | undefined;

  openai: OpenAIApi;

  constructor(
    openAiConfiguration: Configuration,
    openAICompletionRequest: CreateCompletionRequest,
    compressionType: CompressionType = "default"
  ) {
    this.openAiConfiguration = openAiConfiguration;
    this.openAICompletionRequest = openAICompletionRequest;
    this.compressionType = compressionType;
    this.openai = new OpenAIApi(openAiConfiguration);

    if (
      this.openAICompletionRequest.prompt === undefined ||
      null ||
      this.openAICompletionRequest.model === undefined ||
      null
    ) {
      throw new Error("Prompt and Model are required");
    }

    if (typeof this.openAICompletionRequest.prompt !== "string") {
      throw new Error("Prompt must be a string.");
    }

    this.originalPrompt = new BasePrompt(
      this.openAICompletionRequest,
      this.openai
    );

    if (compressionType !== "none") {
      const dietPrompt = this.compressPrompt(
        this.openAICompletionRequest.prompt
      );
      this.openAICompletionRequest.prompt = dietPrompt;
      this.dietPrompt = new BasePrompt(
        this.openAICompletionRequest,
        this.openai
      );
    }
  }

  originalRequest() {
    return this.openAICompletionRequest;
  }

  compressPrompt(prompt: string) {
    switch (this.compressionType) {
      case "default":
        return compressPromptDefault(prompt);
    }
  }
}

const PUNCTUATION = [".", ",", "'", '"', "!", "?", ";", ":", "-"];

const compressPromptDefault = (prompt: string) => {
  prompt = prompt.replace(/do not/gi, "don't");

  // remove punctuation
  prompt = prompt
    .split("")
    .filter((char) => !PUNCTUATION.includes(char))
    .join("");

  // remove stopwords
  prompt = prompt
    .split(" ")
    .filter((word) => !stopwords.english.includes(word))
    .join(" ");

  return prompt;
};

class BasePrompt {
  request: CreateCompletionRequest;
  tiktokenEncoding: Tiktoken;
  openai: OpenAIApi;

  constructor(request: CreateCompletionRequest, openai: OpenAIApi) {
    this.tiktokenEncoding = encoding_for_model(request.model as TiktokenModel);
    this.request = request;
    this.openai = openai;
  }

  estimatedPromptCost() {
    return this.request.prompt.length;
  }

  tokensCount() {
    const tokens = this.tiktokenEncoding.encode(this.request.prompt as string);
    this.tiktokenEncoding.free();
    return tokens.length;
  }

  tokens() {
    const tokens = this.tiktokenEncoding.encode(this.request.prompt as string);
    this.tiktokenEncoding.free();
    return tokens;
  }

  getPrompt() {
    return this.request.prompt;
  }
}

console.log(
  compressPromptDefault(
    "I want you to act as an advertiser. You will create a campaign to promote a product or service of your choice. You will choose a target audience, develop key messages and slogans, select the media channels for promotion, and decide on any additional activities needed to reach your goals. My first suggestion request is 'I need help creating an advertising campaign for a new type of energy drink targeting young adults aged 18-30.'"
  )
);
