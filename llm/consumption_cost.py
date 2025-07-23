def estimate_consumption(seqlen, batch_size, hidden_size, attention_heads):
  #34 sbh + 5as²b
  return round((34*seqlen*batch_size*hidden_size + 5*attention_heads*seqlen*seqlen*batch_size)*2/(1024**3),2)

def estimate_optimizer_size(nb_billion_parameters, bitwidth_optimizer):
  return round((2*nb_billion_parameters*bitwidth_optimizer/8*(1000**3))/(1024**3),2)

def estimate_model_size(nb_billion_parameters, bitwidth_model):
  return round(nb_billion_parameters*bitwidth_model/8*(1000**3)/(1024**3),2)

nb_billion_parameters = 1.23
bitwidth_model = 32
bitwidth_optimiser = 32 # 32? 
seqlen = 256
batch_size = 8
hidden_layers = 16
hidden_size = 2048
attention_heads = 32

activation_consumption = estimate_consumption(seqlen, batch_size, hidden_size, attention_heads)
model_consumption = estimate_model_size(nb_billion_parameters, bitwidth_model)
optimizer_consumption = estimate_optimizer_size(nb_billion_parameters, bitwidth_optimiser)

print("Memory consumption of the model: "+str(model_consumption)+" GB\n")

print("Memory consumption of the optimizer: "+str(optimizer_consumption)+" GB")
print("Memory consumption of activations for fine-tuning: "+str(activation_consumption*hidden_layers)+" GB")
print("Total memory consumption for fine-tuning: "+str(model_consumption+optimizer_consumption+activation_consumption*hidden_layers)+" GB\n")

print("Memory consumption of activations for inference: "+str(activation_consumption)+" GB")
print("Total memory consumption for inference: "+str(model_consumption+activation_consumption)+" GB")
