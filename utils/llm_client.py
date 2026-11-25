import os
import time
from dataclasses import dataclass
from google import genai
from google.genai import types
from groq import Groq


@dataclass
class LLMResponse:
    provider: str
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_sec: float


class LLMClient:
    """Wrapper terpadu untuk Gemini API dan Groq API (temperature=0 + backoff)."""

    MODEL_MAP = {
        "gemini": "gemini-3.8-flash",
        "groq": "openai/gpt-oss-120b",
    }

    def __init__(self, provider: str = "groq", max_retries: int = 3):
        if provider not in self.MODEL_MAP:
            raise ValueError(f"Provider tidak dikenal: {provider}. Pilih 'gemini' atau 'groq'.")
        self.provider = provider
        self.model = self.MODEL_MAP[provider]
        self.max_retries = max_retries

        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY belum diatur di environment/secrets.")
            self.client = genai.Client(api_key=api_key)
        else:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY belum diatur di environment/secrets.")
            self.client = Groq(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        last_error = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                if self.provider == "gemini":
                    return self._call_gemini(system_prompt, user_prompt, start)
                return self._call_groq(system_prompt, user_prompt, start)
            except Exception as exc:
                last_error = exc
                wait_sec = 2 ** (attempt + 1)
                print(f"[{self.provider}] Percobaan {attempt + 1} gagal ({exc}). Menunggu {wait_sec}s...")
                time.sleep(wait_sec)
        raise RuntimeError(f"Gagal memanggil {self.provider} setelah {self.max_retries} kali: {last_error}")

    def _call_gemini(self, system_prompt: str, user_prompt: str, start: float) -> LLMResponse:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
            ),
        )
        latency = round(time.perf_counter() - start, 3)
        usage = getattr(resp, "usage_metadata", None)
        in_tok = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_tok = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        tot_tok = (getattr(usage, "total_token_count", 0) or (in_tok + out_tok)) if usage else (in_tok + out_tok)

        return LLMResponse(
            provider=self.provider,
            model=self.model,
            content=(resp.text or "").strip(),
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            latency_sec=latency,
        )

    def _call_groq(self, system_prompt: str, user_prompt: str, start: float) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        latency = round(time.perf_counter() - start, 3)
        content = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        tot_tok = usage.total_tokens if usage else (in_tok + out_tok)

        return LLMResponse(
            provider=self.provider,
            model=self.model,
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            latency_sec=latency,
        )


if __name__ == "__main__":
    sys_msg = "Kamu adalah peninjau kode Python. Jawab singkat dalam 1 kalimat."
    usr_msg = "Apakah fungsi `def add(a, b): return a - b` sudah benar untuk menjumlahkan dua angka?"

    for prov in ("gemini", "groq"):
        print(f"\n=== Menguji {prov.upper()} ===")
        try:
            client = LLMClient(provider=prov)
            res = client.generate(sys_msg, usr_msg)
            print(f"Model   : {res.model}")
            print(f"Jawaban : {res.content}")
            print(f"Token   : input={res.input_tokens}, output={res.output_tokens}, total={res.total_tokens}")
            print(f"Waktu   : {res.latency_sec} detik")
        except Exception as e:
            print(f"ERROR   : {e}")
