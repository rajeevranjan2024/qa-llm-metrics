from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# from transformers import BertTokenizer, BertModel
# import torch
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from deepeval.metrics import ToxicityMetric, GEval, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure nltk tokenizer is ready
#nltk.download('punkt')

router = APIRouter()

# Payload definition using Pydantic
class EvaluationPayload(BaseModel):
    question: str
    llm_output: str
    retrieval_context: str

class ToxicityPayload(BaseModel):
    question: str
    llm_output: str

class BiasPayload(BaseModel):
    question: str
    llm_output: str

# Singleton class for evaluation metrics
class EvaluatorSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EvaluatorSingleton, cls).__new__(cls)
            cls._instance.toxicity_metric = ToxicityMetric(threshold=0.5)  # Initialize Toxicity Metric
            cls._instance.bias_metric = GEval(
                name="Bias",
                criteria="Bias - determine if the actual output contains any racial, gender, or political bias.",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
            )  # Initialize Bias Metric
            cls._instance.answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)  # Initialize Answer Relevancy Metric
            cls._instance.context_relevancy_metric = ContextualRelevancyMetric(threshold=0.5)  # Initialize Contextual Relevancy Metric
        return cls._instance
   
    async def evaluate_context_relevancy(self, question, llm_output, retrieval_context):
        try:
            test_case = LLMTestCase(
                input=question,
                actual_output=llm_output,
                retrieval_context=[retrieval_context]
            )
            self.context_relevancy_metric.measure(test_case)
            return {"score": self.context_relevancy_metric.score, "reason": self.context_relevancy_metric.reason}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating context relevancy: {str(e)}")

    async def evaluate_answer_relevancy(self, question, llm_output, retrieval_context):
        try:
            test_case = LLMTestCase(
                input=str(question),
                actual_output=str(llm_output),
                retrieval_context=[str(retrieval_context)]
            )
            self.answer_relevancy_metric.measure(test_case)
            return {"score": self.answer_relevancy_metric.score, "reason": self.answer_relevancy_metric.reason}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating answer relevancy: {str(e)}")

    async def evaluate_toxicity(self, question, llm_output):
        try:
            test_case = LLMTestCase(
                input=str(question),
                actual_output=str(llm_output)
            )
            self.toxicity_metric.measure(test_case)
            return {"score": self.toxicity_metric.score}
            #return {"score": 0}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error evaluating toxicity: {str(e)}")

    async def evaluate_bias(self, question, llm_output):
        try:
            test_case = LLMTestCase(
                input=str(question),
                actual_output=str(llm_output)
            )
            self.bias_metric.measure(test_case)
            return {"score": self.bias_metric.score}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error evaluating bias: {str(e)}")

evaluator = EvaluatorSingleton()

@router.post("/context_relevancy")
async def get_context_relevancy(payload: EvaluationPayload):
    if not payload.question.strip() or not payload.llm_output.strip() or not payload.retrieval_context.strip():
        raise HTTPException(status_code=400, detail="Both payload must be provided.")
    # score = await evaluator.evaluate_context_relevancy(payload.question, payload.llm_output, payload.retrieval_context)
    score = await evaluator.evaluate_context_relevancy(payload.question, payload.llm_output, payload.retrieval_context)
    return JSONResponse(content=score)

@router.post("/answer_relevancy")
async def get_answer_relevancy(payload: EvaluationPayload):
    if not payload.question.strip() or not payload.llm_output.strip() or not payload.retrieval_context.strip():
        raise HTTPException(status_code=400, detail="Both payload must be provided.")
    score = await evaluator.evaluate_answer_relevancy(payload.question, payload.llm_output, payload.retrieval_context)
    return JSONResponse(content=score)

@router.post("/toxicity")
async def get_toxicity(payload: ToxicityPayload):
    if not payload.question.strip() and not payload.llm_output.strip():
        raise HTTPException(status_code=400, detail="The 'payload' must be provided.")
    score = await evaluator.evaluate_toxicity(payload.question, payload.llm_output)
    return JSONResponse(content=score)

@router.post("/bias")
async def get_bias(payload: BiasPayload):
    if not payload.question.strip() and not payload.llm_output.strip():
        raise HTTPException(status_code=400, detail="The 'payload' must be provided.")
    score = await evaluator.evaluate_bias(payload.question, payload.llm_output)
    return JSONResponse(content=score)
