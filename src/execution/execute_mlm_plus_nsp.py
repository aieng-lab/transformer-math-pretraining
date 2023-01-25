from src.execution.Executor import Executor
from src.training.training_data.DataEnum import DataEnum
from src.config.params import ScriptParamHandler
from src.pretraining_methods.Objectives import Objectives

if __name__ == "__main__":
    print("\nPYTHON OUTPUT")
    print("-" * 60)
    print("\n")

    param_handler = ScriptParamHandler("--wiki_file", "--book_file", "--local_tokenizer")
    wiki_file = param_handler.get_arg_value("wiki_file")
    book_file = param_handler.get_arg_value("book_file")
    print(f"Wikipedia file: {wiki_file}")
    print(f"Books file: {book_file}")

    runner = Executor()
    runner.initialise(param_handler)

    training_data, validation_data = runner.prepare_training_data([DataEnum.BOOKCORPUS, DataEnum.WIKIPEDIA],
                                                                  file_names=[book_file, wiki_file],
                                                                  column_names=["text"])

    # todo: code for loading custom tokenizer once it exists

    tokenizer = runner.get_huggingface_tokenizer("bert-base-cased")

    trainer, n_train, n_eval = runner.prepare_pretraining([Objectives.MLM, Objectives.NSP], ds_train=training_data,
                                                          ds_val=validation_data, tokenizer=tokenizer)
    trainer.train(n_train, n_eval, with_grad_clipping=True)
