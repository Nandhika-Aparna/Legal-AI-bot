import asyncio
import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from openai import OpenAI
from datasets import Dataset
from typing import List, Dict

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")
openai_client = OpenAI()

# --- Advanced RAG Pipeline Functions ---

def hybrid_search(query: str, top_n: int = 5) -> Dict[str, str]:
    """
    Performs a hybrid search using both semantic and keyword search.
    This function simulates a complex search over a pre-indexed document store.
    
    For this evaluation, we'll use a simplified mock function that
    retrieves documents based on a direct match to the query.
    In a real system, this would involve a vector database and a search index.
    """
    # This is a mock implementation. In a real application, you would connect to a database.
    mock_document_store = {
        "What are the key differences between a lease and a licence?": [
            "Tenant (Fourth Edition)- \"A lease, because it confers an estate in land, is much more than a mere\npersonal or contractual agreement for the occupation of a freeholder's land by a tenant. A lease,\nwhether fixed-term or periodic, confers a right in property, enabling the tenant to exclude all third\nparties, including the landlord, from possession, for the duration of the lease, in return for which a\nrent or periodical payment is reserved out of the land. A contractual licence confers no more than a\npermission on the occupier to do some act on the owner's land which would otherwise constitute a\ntrespass. If exclusive possession is not conferred by an agreement, it is a licence.\"\"\".....the"
        ],
        "If a contract is missing a signature from one of the parties, is it legally binding?": [
            "the lack of signature of one of the contracting parties renders the agreement null and void. What the\nSeth Banarsi Das vs The Cane Commissioner & Another on 6 December, 1962\nIndian Kanoon - http://indiankanoon.org/doc/1116381/ 12",
            "the appellant who is complaining of the want of signature. In our opinion, the agreement was\nbinding. It may be pointed out that the arbitration clause in the agreement was enforceable, if\nagreed to, even without the signature of the appellant as it is settled law that to constitute an\narbitration agreement in writing it is not necessary that it should be signed by the parties and it is\nsufficient if the terms are reduced to writing and the agreement of the parties thereto is established.\nSee Jugal Kishore Rameshwardas v. Mrs. Goolbai Hormusji (1). In our opinion even if the section be\nheld to be mandatory to the extent that the terms as prescribed should appear in writing, that is\ncomplied with in this case. There was thug a binding contract between the parties and the dispute\nwas to be ,resolved as required by Rule 23."
        ],
        "What are the requirements for a valid will in this jurisdiction?": [
            "13. The need and necessity for stringent requirements of clause (c) to Section 63 of the Indian\nSuccession Act has been elucidated and explained in several decisions. In H. Venkatachala Iyengar\nv. B.N. Thimmajamma and Others.2 dilating on the statutory and mandatory requisites for\nvalidating the execution of the Will, this Court had highlighted the dissimilarities between the Will\nwhich is a testamentary instrument vis-„-vis other documents of conveyancing, by emphasising that\nthe Will is produced before the court after the testator who has departed from the world, cannot say\nthat the Will is his own or it is not the same. This factum introduces an element of solemnity to the\ndecision on the question where the Will propounded is proved as the last Will or testament of the\ndeparted testator. Therefore, the propounder to succeed and prove the Will is required to prove by\nsatisfactory evidence that (i) the Will was signed by the testator; (ii) the testator at the time was in a",
            "concerned in this appeal, is that the will has to be attested by two or more witnesses and each of\nthese witnesses must have seen the testator sign or affix his mark to the Will, or must have seen\nsome other person sign the Will in the presence and by the direction of the testator, or must have\nreceived from the testator a personal acknowledgement of signature or mark, or of the signature of\nsuch other person, and each of the witnesses has to sign the Will in the presence of the testator.\nIt is thus clear that one of the requirements of due execution of will is its attestation by two or more\nwitnesses which is mandatory."
        ]
    }
    return {query: mock_document_store.get(query, [])}

def rerank_documents(query: str, documents: List[str], top_n: int = 5) -> List[str]:
    """
    Reranks a list of documents based on their relevance to a query.
    For this evaluation, we use a simple mock to return the documents as-is.
    In a real system, you would use a dedicated reranker model.
    """
    print("Re-ranked Contexts:")
    for doc in documents:
        print(doc[:150] + "...") # Print a snippet for brevity
    print("---------------------------")
    return documents[:top_n]

def summarize_contexts(client: OpenAI, query: str, contexts: List[str]) -> str:
    """
    Summarizes the provided documents into a concise, factual string.
    This function acts as a pre-processing step for the final LLM call.
    """
    context_string = "\n\n---\n\n".join(contexts)
    messages = [
        {
            "role": "system",
            "content": """You are an expert legal fact extractor.
Your task is to analyze the provided legal documents and extract all facts and key concepts that are directly relevant to the user's question.
Present the extracted information as a bulleted list.
Do not add any opinions, analysis, or information not present in the documents.
If no relevant information is found, state "No relevant facts found."
"""
        },
        {"role": "user", "content": f"Query: {query}\n\nDocuments:\n{context_string}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

def generate_final_answer(client: OpenAI, user_query: str, summarized_context: str) -> str:
    """
    Generates a final, high-quality response based on a summarized context.
    This function represents the final LLM call in the pipeline.
    """
    messages = [
        {
            "role": "system",
            "content": f"""You are a meticulous legal expert. Your task is to provide a final, high-quality, and complete response to the user's query.

Follow these instructions strictly to achieve high faithfulness:
1.  **Directly address the Original Query.**
2.  **Use ONLY the provided Context.**
3.  **Be concise and accurate.**
4.  If the context contains contradictory or nuanced information, state that the answer is not a simple yes/no and explain the different scenarios described in the context.
5.  If the context does not contain the answer, state "I'm sorry, I cannot answer this question based on the provided documents."

Original Query: {user_query}

Context:
{summarized_context}
"""
        },
        {"role": "user", "content": f"Based on the context, provide the final, complete answer to the query: {user_query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

# --- RAGAS Evaluation Loop ---

async def run_evaluation():
    # Use a small, hardcoded dataset for demonstration
    test_cases = [
        "What are the key differences between a lease and a licence?",
        "If a contract is missing a signature from one of the parties, is it legally binding?",
        "What are the requirements for a valid will in this jurisdiction?"
    ]

    questions = []
    contexts = []
    answers = []

    for user_input in test_cases:
        print(f"Processing query: {user_input}")

        # Step 1: Hybrid Search
        retrieved_contexts_dict = hybrid_search(user_input)
        retrieved_contexts = list(retrieved_contexts_dict.values())[0]
        
        # Step 2: Re-rank documents
        final_contexts_list = rerank_documents(user_input, retrieved_contexts, top_n=5)
        
        # Step 3: Summarize contexts
        if final_contexts_list:
            summarized_context = summarize_contexts(openai_client, user_input, final_contexts_list)
        else:
            summarized_context = "No relevant documents were found."
            
        print("\nSummarized Context for Final Answer:")
        print(summarized_context)
        print("--------------------------------------\n")

        # Step 4: Generate the final answer
        final_answer = generate_final_answer(openai_client, user_input, summarized_context)

        print("\nFinal Answer:")
        print(final_answer)
        print("---------------------------\n")

        questions.append(user_input)
        contexts.append(final_contexts_list)
        answers.append(final_answer)

    # Convert lists to a Ragas-compatible dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers
    })

    # Run Ragas evaluation
    print("Running Ragas evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ]
    )

    print("\n--- Ragas Evaluation Results ---")
    print(result)
    print("\n--- Detailed Results ---")
    df = result.to_pandas()
    # Corrected column names for the Ragas DataFrame output
    print(df)

if __name__ == "__main__":
    # Ensure you have your API keys set up as environment variables
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    asyncio.run(run_evaluation())