
def batch_predict(model, batch_inputs):
    model.eval()
    batch_outputs = model(batch_inputs)
    return batch_outputs.detach().float().cpu().numpy()
