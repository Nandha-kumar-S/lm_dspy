import dspy
from typing import Literal

#define Signature
class Astronautics_QA(dspy.Signature):
    """Answer the question using astronautics knowledge"""
    question = dspy.InputField(desc="question based on astronautics")
    choices = dspy.InputField(desc="list of multiple choices for the question")
    answer = dspy.OutputField(desc="should best a matching answer from choices")

#define Module
class Classification(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(Astronautics_QA)
    
    def forward(self, question, choices):
        response = self.cot(question=question, choices=choices)
        return response