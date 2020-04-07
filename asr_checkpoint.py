import torch

from data_related.audio_feature_extraction import AudioFeaturesConfig
from data_related.char_stt_dataset import DataConfig
from data_related.librispeech import LIBRI_VOCAB
from model import DeepSpeech


def build_checkpoint_package(
        model:DeepSpeech,
        optimizer=None,
        amp=None,
        epoch=None,
        iteration=None,
        log_data=None,
        avg_loss=None,
        meta=None,
        data_conf:DataConfig=None,
        audio_conf:AudioFeaturesConfig=None,
):
    from apex.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        model = model.module

    model: DeepSpeech

    package = {
        "version": model.version,
        "hidden_size": model.hidden_size,
        "hidden_layers": model.hidden_layers,
        "vocab_size": model.vocab_size,
        "input_feature_dim": model.input_feature_dim,
        "state_dict": model.state_dict(),
        "bidirectional": model.bidirectional,
        "data_conf":data_conf,
        "audio_conf":audio_conf
    }
    if optimizer is not None:
        package["optim_dict"] = optimizer.state_dict()
    if amp is not None:
        package["amp"] = amp.state_dict()
    if avg_loss is not None:
        package["avg_loss"] = avg_loss
    if epoch is not None:
        package["epoch"] = epoch + 1  # increment for readability
    if iteration is not None:
        package["iteration"] = iteration
    if log_data is not None:
        package.update(log_data)
    if meta is not None:
        package["meta"] = meta
    return package

def load_trainable_checkpoint(file):
    package = torch.load(
        file, map_location=lambda storage, loc: storage
    )
    model = DeepSpeech.load_model_package(package)

    return package,model

def load_evaluatable_checkpoint(device, file, use_half):
    package = torch.load(file, map_location=lambda storage, loc: storage)
    package['rnn_hidden_size'] = package[
        'hidden_size']  # TODO(tilo):backward compatibility

    model: DeepSpeech = DeepSpeech.load_model(package)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()

    data_conf = package.get('data_conf', DataConfig(LIBRI_VOCAB))#TODO(tilo): only for backwards compatibility
    audio_conf = package.get('audio_conf', AudioFeaturesConfig())#TODO(tilo): only for backwards compatibility

    return model,data_conf,audio_conf