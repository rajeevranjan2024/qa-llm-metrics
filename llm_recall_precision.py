from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from deepeval.metrics import ToxicityMetric, GEval, AnswerRelevancyMetric, ContextualRelevancyMetric, ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Define a router for the endpoints
router = APIRouter()

# Payload definition using Pydantic
class EvaluationPayload(BaseModel):
    question: str
    llm_output: str
    retrieval_context: str

class ContextualPayload(BaseModel):
    question: str
    llm_output: str
    ground_truth: str  # Include expected or ideal output for contextual metrics
    retrieval_context: str

# Singleton class for evaluation metrics
class EvaluatorSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EvaluatorSingleton, cls).__new__(cls)
            cls._instance.contextual_recall_metric = ContextualRecallMetric(threshold=0.5)  # Initialize Contextual Recall Metric
            cls._instance.contextual_precision_metric = ContextualPrecisionMetric(threshold=0.5)  # Initialize Contextual Precision Metric
        return cls._instance

    async def evaluate_contextual_recall(self, question, llm_output, ground_truth, retrieval_context):
        try:
            test_case = LLMTestCase(
                input=question,
                actual_output=llm_output,
                expected_output=ground_truth,
                retrieval_context=[retrieval_context]
            )
            self.contextual_recall_metric.measure(test_case)
            return {"score": self.contextual_recall_metric.score, "reason": self.contextual_recall_metric.reason}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating contextual recall: {str(e)}")

    async def evaluate_contextual_precision(self, question, llm_output, ground_truth, retrieval_context):
        try:
            test_case = LLMTestCase(
                input=question,
                actual_output=llm_output,
                expected_output=ground_truth,
                retrieval_context=[retrieval_context]
            )
            self.contextual_precision_metric.measure(test_case)
            return {"score": self.contextual_precision_metric.score, "reason": self.contextual_precision_metric.reason}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating contextual precision: {str(e)}")

evaluator = EvaluatorSingleton()

# Define routes for contextual recall and precision
@router.post("/context_recall")
async def get_context_recall(payload: ContextualPayload):
    if not payload.question.strip() or not payload.llm_output.strip() or not payload.ground_truth.strip() or not payload.retrieval_context.strip():
        raise HTTPException(status_code=400, detail="All fields must be provided.")
    score = await evaluator.evaluate_contextual_recall(payload.question, payload.llm_output, payload.ground_truth, payload.retrieval_context)
    return JSONResponse(content=score)

@router.post("/context_precision")
async def get_context_precision(payload: ContextualPayload):
    if not payload.question.strip() or not payload.llm_output.strip() or not payload.ground_truth.strip() or not payload.retrieval_context.strip():
        raise HTTPException(status_code=400, detail="All fields must be provided.")
    score = await evaluator.evaluate_contextual_precision(payload.question, payload.llm_output, payload.ground_truth, payload.retrieval_context)
    return JSONResponse(content=score)
