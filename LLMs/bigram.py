"""
Bigram Language Model (BLM)
"""

from typing import Callable, List, Union
import torch


class BigramLanguageModel:
    """Bigram Language Model class."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def __str__(self) -> str:
        return "BigramLanguageModel instance"

    def __repr__(self) -> str:
        return "BigramLanguageModel()"

    def set_device(self, device_str: str) -> None:
        """
        Set the device to 'cpu' or 'cuda'.

        Args:
            device_str (str): Device string, 'cpu' or 'cuda'.
        """
        if device_str not in ("cpu", "cuda"):
            raise ValueError("Device must be 'cpu' or 'cuda'")
        if device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.device = torch.device(device_str)

    def get_device(self) -> torch.device:
        """
        Get the current device.

        Returns:
            torch.device: The current device.
        """
        return self.device

    def open_txt_file(self, path_to_file: str) -> str:
        """Open and read text from a .txt file."""
        if not path_to_file.endswith(".txt"):
            raise ValueError("File must be a `.txt` file")

        try:
            with open(path_to_file, "r", encoding="utf-8") as file:
                text = file.read()
            return text
        except Exception as exc:
            raise IOError(f"Failed to read the file: {exc}")

    def get_unique_characters_in_text(self, text: str) -> List[str]:
        """Get a sorted list of unique characters in text."""
        if not isinstance(text, str) or not text:
            raise ValueError("Input should be a non-empty string")

        return sorted(set(text))

    def character_level_tokeniser(
        self,
        char_list: List[str],
        decode: bool = True,
        encoder: Callable = None,
        decoder: Callable = None,
    ) -> Union[List[int], str]:
        """Encode or decode characters using encoder/decoder functions."""
        if (
            not char_list
            or not isinstance(char_list, list)
            or not all(isinstance(c, str) and len(c) == 1 for c in char_list)
        ):
            raise ValueError("Input should be a list of single-character strings")

        if decode:
            if decoder is None:
                raise ValueError("Decoder function must be provided for decoding")
            return decoder(char_list)
        else:
            if encoder is None:
                raise ValueError("Encoder function must be provided for encoding")
            return encoder(char_list)

    def character_level_encoder(
        self, char_list: List[str]
    ) -> Callable[[str], List[int]]:
        """Return function to encode string to list of ints based on char list."""
        if not char_list or not isinstance(char_list, list):
            raise ValueError("Input should be a list of characters")

        char_to_int = {ch: i for i, ch in enumerate(char_list)}

        def encode(s: str) -> List[int]:
            return [char_to_int[c] for c in s]

        return encode

    def character_level_decoder(
        self, char_list: List[str]
    ) -> Callable[[List[int]], str]:
        """Return function to decode list of ints to string based on char list."""
        if not char_list or not isinstance(char_list, list):
            raise ValueError("Input should be a list of characters")

        int_to_char = {i: ch for i, ch in enumerate(char_list)}

        def decode(l: List[int]) -> str:
            return "".join(int_to_char[i] for i in l)

        return decode

    def get_data_in_tensor_format(self, path_to_file: str) -> torch.Tensor:
        """Load text file and return tensor of encoded characters on the device."""
        text = self.open_txt_file(path_to_file)
        if not text:
            raise ValueError("File is empty or text is invalid")

        unique_chars = self.get_unique_characters_in_text(text)
        encoder = self.character_level_encoder(unique_chars)
        encoded_text = encoder(text)
        data = torch.tensor(encoded_text, dtype=torch.long, device=self.device)

        return data

    def get_train_and_validation_splits(
        self, tensor: torch.Tensor, split_ratio: float = 0.9
    ) -> List[torch.Tensor]:
        """Split tensor data into training and validation sets."""
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Argument must be a torch.Tensor")

        n = int(split_ratio * len(tensor))
        train_data = tensor[:n].to(self.device)
        val_data = tensor[n:].to(self.device)

        return [train_data, val_data]
