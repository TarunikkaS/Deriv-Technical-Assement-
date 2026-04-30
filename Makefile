.PHONY: run cli validate install clean

install:
	pip install -r requirements.txt

run:
	python run_pipeline.py

cli:
	python -m src.cli

validate:
	python validate.py

clean:
	rm -rf artifacts/raw_pages artifacts/cleaned_pages.json \
	       artifacts/corpus.json artifacts/vector_store \
	       artifacts/retrieval_logs.jsonl artifacts/generated_answers.json \
	       artifacts/grounding_verification.json artifacts/answer_audit.json \
	       artifacts/llm_calls.jsonl artifacts/answer_quality_scores.json \
	       artifacts/knowledge_gap_report.json artifacts/corpus_version_report.json \
	       artifacts/validation_report.json
