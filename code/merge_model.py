from transformers import(AutoTokenizer,
                         AutoModelForCausalLM,
                         BitsAndBytesConfig,
                         HfArgumentParser
                         )
from peft import (PeftModel,
                  PeftConfig)
import bitsandbytes
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MergingArguments:
    base_model: Optional[str] = field(
        default=None, metadata={"help" : "The name of the model."}
    )
    peft_model: Optional[str] = field(
        default=None, metadata={"help" : "The name of the finetuned model."}
    )
    
def main():
    parser = HfArgumentParser((MergingArguments))
    merging_arguments  = parser.parse_args_into_dataclasses()[0]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    config = PeftConfig.from_pretrained(merging_arguments.peft_model)
    model = AutoModelForCausalLM.from_pretrained(merging_arguments.base_model,
                                                 device_map="cuda:0",
                                                 quantization_config=bnb_config)
    peft_model = PeftModel.from_pretrained(model, merging_arguments.peft_model)
    tokenizer = AutoTokenizer.from_pretrained(merging_arguments.peft_model)
    
    
    merged_model = peft_model.merge_and_unload() 
    merged_model.push_to_hub(f'''{merging_arguments.peft_model}_merged''')
    tokenizer.push_to_hub(f'''{merging_arguments.peft_model}_merged''')    

if __name__ == "__main__":main()
