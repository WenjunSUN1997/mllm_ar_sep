import torch

from mplug_owl.modeling_mplug_owl import *
from transformers import AutoModel

class MplugOwlForTokenClassification(MplugOwlForConditionalGeneration):
    def __init__(self, config: MplugOwlConfig, weight_loss, weight_flag):
        super(MplugOwlForTokenClassification, self).__init__(config)
        language_model = AutoModel.from_config(config.text_config)
        self.language_model = language_model
        self.linear = torch.nn.Linear(in_features=4096, out_features=len(weight_loss))
        weight_loss = torch.tensor(weight_loss).type(self.linear.weight.dtype).to(self.linear.weight.device)
        if weight_flag:
            self.loss_func = torch.nn.CrossEntropyLoss(weight=weight_loss)
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()

    def _forward(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the mPLUG-Owl2 as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.vision_model.embeddings.cls_token.data.dtype)

        if input_ids is None:
            raise EOFError

        if attention_mask is None:
            attention_mask = input_ids.new_ones(*input_ids.shape)

        batch_size = input_ids.size(0)
        pad_tensor = torch.tensor([-1]*65).to(input_ids.device).unsqueeze(0)
        input_ids = torch.cat((input_ids, pad_tensor), dim=-1)
        media_token_indices = [get_media_indices(input_ids[i]) for i in range(batch_size)]
        num_images_per_sample = [len(x) for x in media_token_indices]
        input_ids = input_ids.clone()  # prevent inplace modify
        input_ids[input_ids < 0] = 0  # Not used

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        batch_size = input_ids.shape[0]
        # get text embedding
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if hasattr(self.language_model, 'transformer') and hasattr(self.language_model.transformer, 'word_embeddings_layernorm'):
            inputs_embeds = self.language_model.transformer.word_embeddings_layernorm(inputs_embeds)
        # get visual embedding
        if pixel_values is not None:
            pixel_values = pixel_values.to(input_ids.device)
            with torch.no_grad():
                image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
                image_attention_mask = torch.ones(
                    image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
                )
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_outputs = self.abstractor(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
                query_output = query_outputs["last_hidden_state"]
                image_embeds = query_output
            img_seq_length = image_embeds.shape[1]

            # ===================
            # Get actual input embeddings
            # ===================
            text_chunk_embeds = []
            text_chunk_attns = []
            img_idx = 0

            for b in range(batch_size):
                start = 0
                result = []
                result_attn = []
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(inputs_embeds[b, start:pos])
                        result_attn.append(attention_mask[b, start:pos])
                    result.append(image_embeds[img_idx + i])
                    result_attn.append(torch.ones(image_embeds[img_idx + i].shape[0], device=inputs_embeds.device))
                    start = pos + img_seq_length
                if start < inputs_embeds.shape[1]:
                    result.append(inputs_embeds[b, start:])
                    result_attn.append(attention_mask[b, start:])

                img_idx += num_images_per_sample[b]
                text_chunk_embeds.append(torch.cat(result, dim=0))
                text_chunk_attns.append(torch.cat(result_attn, dim=0))
            inputs_embeds = torch.stack(text_chunk_embeds, dim=0)
            attention_mask = torch.stack(text_chunk_attns, dim=0)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            # input_ids=input_ids,
            attention_mask=attention_mask
        )

        return outputs

    def forward(self, **data):
        loss_all = []
        predicition_all = []
        batch_size = data['word_ids'].shape[0]
        for batch_index in range(batch_size):
            input_this_batch = self.process_per_batch(data, batch_index)
            semantic_this_batch = self._forward(pixel_values=input_this_batch['pixel_values'],
                                                input_ids=input_this_batch['input_ids'],
                                                attention_mask=input_this_batch['attention_mask'])['last_hidden_state']
            first_token_semantic = semantic_this_batch[0][input_this_batch['first_token_index']]
            output_linear = self.linear(first_token_semantic)
            predicition = torch.max(output_linear, dim=-1).indices.cpu().numpy().tolist()
            loss = self.loss_func(output_linear, input_this_batch['label'])
            loss_all.append(loss)
            predicition_all += predicition

        return {'loss': sum(loss_all),
                'prediction': predicition_all}

    def process_per_batch(self, data, batch_index):
        input_ids = data['input_ids'][batch_index]
        pixel_values = data['pixel_values'][batch_index]
        attention_mask = data['attention_mask'][batch_index]
        word_ids = data['word_ids'][batch_index]
        label = data['label'][batch_index]
        real_token_index = attention_mask == 1
        input_ids = input_ids[real_token_index]
        attention_mask = attention_mask[real_token_index]
        word_ids = word_ids[real_token_index]
        label = label[label != -100]
        first_token_index = self.get_first_token_index(word_ids)
        assert len(first_token_index) == len(label)
        result = {'input_ids': input_ids.unsqueeze(0),
                  'attention_mask': attention_mask.unsqueeze(0),
                  'label': label,
                  'first_token_index': first_token_index,
                  'pixel_values': pixel_values.unsqueeze(0)}
        return result

    def get_first_token_index(self, word_ids):
        word_id_current = -100
        result = []
        for token_index in range(len(word_ids)):
            if word_ids[token_index] != word_id_current:
                result.append(token_index)
                word_id_current = word_ids[token_index]

        return result

