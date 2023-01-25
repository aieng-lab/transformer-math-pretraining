from training.training_data import TrainingData
from execution.Executor import Executor
from src.pretraining_methods.pretraining_funcs import mlm_plus_nsp
from src.models.pretraining_model import PretrainingModel

from transformers import BertTokenizerFast
import torch

if __name__ == '__main__':
    print("\n")
    print("-" * 50)
    print("PYTHON OUTPUT")
    print("-" * 50, "\n")

    train_data = TrainingData("books", "wiki/20220301.en")
    train_data.show_ds_info()
    train_data.set_data_path("/home/katja/singularity/python-images/transformer_pretraining/python/train_data")

    ds_books = train_data.load_training_data("books", percentage=5, save=True)
    ds_wiki = train_data.load_training_data("wiki", percentage=5, save=True)
    ds_wiki = train_data.process_wiki_sentence_level(ds_wiki, save=True)
    ds_wiki = ds_wiki.remove_columns([name for name in ds_wiki.column_names if name != "text"])
    training_data = train_data.concat_datasets(ds_books, ds_wiki)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")



    executer = Executor()
    executer.config.BATCH_SIZE = 1
    executer.config.DATA_WORKERS = 1
    dataloader = executer.create_train_data_loader(training_data, tokenizer, 128, mlm_plus_nsp)

    model = PretrainingModel("MLM", "NSP")

    with torch.no_grad():
        model = model.eval()
        for d in dataloader:
            input_ids = d["bert_input"]
            attention_mask = d["attention_mask"]
            segment_ids = d["segment_label"]
            mlm_label = d["bert_label_mlm"]
            nsp_label = d["bert_label_nsp"]
            mask_positions = d["bert_label_mask_positions"]

            mlm_outputs, nsp_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                             segment_ids=segment_ids)
            _, mlm_predictions = torch.max(mlm_outputs, dim=-1)

            mlm_predicted_tokens = tokenizer.convert_ids_to_tokens(mlm_predictions.flatten())

            print("\n")
            print(mlm_predicted_tokens)



            break
