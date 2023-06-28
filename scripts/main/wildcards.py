import os, copy
from pathlib import Path
import msgspec
msgspec_encoders = {".json": msgspec.json.Encoder(), ".msgpack": msgspec.msgpack.Encoder(), ".toml": msgspec.toml, ".yaml": msgspec.yaml}
msgspec_decoders = {".json": msgspec.json.Decoder(), ".msgpack": msgspec.msgpack.Decoder(), ".toml": msgspec.toml, ".yaml": msgspec.yaml}

from dynamicprompts.generators import RandomPromptGenerator, CombinatorialPromptGenerator, JinjaGenerator
from dynamicprompts.generators.feelinglucky import FeelingLuckyGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager

from modules.scripts import basedir

class CardDealer():
    def __init__(self):
        self.__location__ = basedir()
        self.wm = WildcardManager(Path(os.path.join(self.__location__, "wildcards")))
        self.random_generator = RandomPromptGenerator(wildcard_manager=self.wm)
        self.combinatorial_generator = CombinatorialPromptGenerator(wildcard_manager=self.wm)
        self.jinja_generator = JinjaGenerator(wildcard_manager=self.wm)
        self.lucky_generator = FeelingLuckyGenerator(self.random_generator)
        self.payloads = {}
    def wildcard_payload(self, payload_path):
        cached_payloads = self.payloads.get(payload_path)
        if cached_payloads is None:
            args = msgspec_decoders[os.path.splitext(payload_path)[1]].decode(open(payload_path, "rb").read())
            wildcard_type, n_iter, prompt, negative_prompt = args.get("wildcard_type"), args["n_iter"], args["prompt"], args["negative_prompt"]
            generator = self.random_generator if wildcard_type == "random" else self.combinatorial_generator if wildcard_type == "combinatorial" else self.jinja_generator if wildcard_type == "jinja2" else self.lucky_generator if wildcard_type == "lucky" else None
            if generator is None:
                self.payloads.update({payload_path: [args]})
            else:
                payloads = []
                for prompt, negative_prompt in zip(generator.generate(prompt, n_iter), generator.generate(negative_prompt, n_iter)):
                    payloads.append({**copy.deepcopy(args), "prompt": prompt, "negative_prompt": negative_prompt, "n_iter": 1})
                self.payloads.update({payload_path: payloads})
        return copy.deepcopy(self.payloads[payload_path])
