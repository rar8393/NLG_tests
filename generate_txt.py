from transformers import GPT2Tokenizer, OPTForCausalLM
import generated_text_metrics as gtm

class generate_model_results:
    def __init__(tokenizer, model):
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        
    def get_embeddings(output):
        '''
        sequence = tokenizer.encode("Here this out. Here", return_tensors="pt")
        embeds = model.get_input_embeddings()(sequence)
        '''
        return model.get_input_embeddings()(output)
        
    def generate_sample(txt, do_sample=True, sample_size=5, max_length=64, visualize=False):
        final_metrics = {}
        input_ids = tokenizer(txt, return_tensors="pt").input_ids #, return_tensors="pt"
        outputs = model.generate(input_ids, do_sample=True, num_return_sequences=sample_size, max_new_tokens=max_length)
        decoded_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_metrics[txt] = {'metric': gtm.generated_text_metric(txt)}
        for i in range(len(decoded_list)):
            generated_txt = decoded_list[i].replace(txt, '')
            final_metrics[generated_txt] = {'metric': gtm.generated_text_metric(generated_txt), 
                                            'sentence_similarity': gtm.generation_sentence_simialrity(txt, generated_txt),
                                            'embedding': get_embeddings(outputs[i]) }
            if visualize:
                gtm.visualize_dependecy_tree(item)
                gtm.visualize_ne_tree(item)
        return final_metrics
    
    def generate_sample_list(txt_list, do_sample=True, sample_size=5, max_length=64, visualize=False):
        result = {}
        for txt in txt_list:
            result['txt'] = generate_sample(txt, do_sample=do_sample, sample_size=sample_size, 
                                            max_length=max_length, visualize=visualize)
        return result