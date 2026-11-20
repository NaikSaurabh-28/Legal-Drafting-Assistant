"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import json

from app import core
from app.clause_classifier import AdapterArtifactError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="legal-drafting-assistant",
        description="Generate legal-style text with n-gram models and classify clauses.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    gen = sub.add_parser("generate", help="draft a sentence, then classify it if the adapter is installed")
    gen.add_argument("--seed", default="", help="words to start the sentence with")
    gen.add_argument("--max-length", type=int, default=25)
    gen.add_argument("--count", type=int, default=1)
    sub.add_parser("evaluate", help="n-gram perplexity and classifier accuracy on held-out data")
    cl = sub.add_parser("classify", help="classify one clause")
    cl.add_argument("text")
    predict = sub.add_parser(
        "predict",
        aliases=["next-word"],
        help="predict likely next tokens from the from-scratch n-gram model",
    )
    predict.add_argument("context", help="preceding words used as prediction context")
    predict.add_argument("--n", type=int, choices=(1, 2, 3), default=3)
    predict.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    if args.command == "generate":
        if args.count < 1:
            parser.error("--count must be at least 1")
        if args.max_length < 1:
            parser.error("--max-length must be at least 1")
    if args.command in {"predict", "next-word"}:
        if not args.context.strip():
            parser.error("prediction context must not be empty")
        if args.top_k < 1:
            parser.error("--top-k must be at least 1")
    if args.command == "classify" and not args.text.strip():
        parser.error("classification text must not be empty")

    if args.command == "hello":
        print("Legal Drafting Assistant is ready. Open README.md for your milestones.")
    elif args.command == "generate":
        generator = core.milestone_2()
        try:
            classifier, model_name = core.load_classifier()
            print(f"Classifier: {model_name}\n")
            for _ in range(args.count):
                out = core.milestone_4(
                    generator, classifier, args.seed, args.max_length
                )
                print(f"Generated: {out['sentence']}")
                if out["classification"]:
                    print(
                        f"Category: {out['classification']['predicted_category']} "
                        f"(confidence {out['classification']['confidence']})"
                    )
        except AdapterArtifactError as error:
            parser.exit(2, f"GPT-2 adapter error: {error}\n")
    elif args.command == "evaluate":
        combined = core.evaluate_all_models()
        from app.evaluation import save_evaluation_report

        output_path = save_evaluation_report(combined)
        for key, report in combined["classifiers"].items():
            if report.get("status") == "not_evaluated":
                print(f"{report['model']}: not evaluated ({report['reason']})")
            else:
                print(
                    f"{report['model']}: test clauses {report['test_size']}, "
                    f"accuracy {report['accuracy']:.3f}"
                )
        print(f"Evaluation report saved to: {output_path}")
        print(json.dumps(combined, indent=2, default=str))
    elif args.command == "classify":
        try:
            classifier, model_name = core.load_classifier()
            print(f"[{model_name}]", classifier.predict(args.text))
        except AdapterArtifactError as error:
            parser.exit(2, f"GPT-2 adapter error: {error}\n")
    elif args.command in {"predict", "next-word"}:
        model = core.milestone_2(n=args.n).model
        context = tuple(core.tokenize(core.clean_text(args.context)))
        for token, probability in model.predict_next(context, top_k=args.top_k):
            print(f"{token}\t{probability:.6f}")


if __name__ == "__main__":
    main()
