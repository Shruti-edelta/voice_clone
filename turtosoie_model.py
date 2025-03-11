from TTS.tts.configs.tortoise_config import TortoiseConfig
from TTS.tts.models.tortoise import Tortoise

config = TortoiseConfig()
model = Tortoise.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="tortoise-tts/", eval=True)


text="This is a beginning of the history. If you want to hear more, please continue."
# with random speaker
# output_dict = model.synthesize(text, config, speaker_id="random", extra_voice_dirs=None)

# cloning a speaker
output_dict = model.synthesize(text, config, speaker_id="tortoise-tts/tortoise/voices/myself", extra_voice_dirs="myself/")