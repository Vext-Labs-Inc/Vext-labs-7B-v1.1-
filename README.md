<p align="center">
  <img src="assets/social-preview.png" alt="Vext Labs 7B" width="700">
</p>

<h1 align="center">Vext-labs-7B-v1.1</h1>

<p align="center">
  <strong>The first open-source language model purpose-built for autonomous penetration testing.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Parameters-7B-green" alt="Parameters">
  <img src="https://img.shields.io/badge/Context-32%2C768_tokens-orange" alt="Context">
  <img src="https://img.shields.io/badge/Precision-bfloat16-purple" alt="Precision">
  <img src="https://img.shields.io/github/stars/Vext-Labs-Inc/Vext-labs-7B-v1.1-?style=social" alt="Stars">
</p>

<p align="center">
  <a href="https://tryvext.com">Website</a> |
  <a href="#quickstart">Quickstart</a> |
  <a href="#what-this-model-does">Capabilities</a> |
  <a href="#benchmarks">Benchmarks</a> |
  <a href="#training">Training</a>
</p>

---

## Overview

**Vext-labs-7B-v1.1** is the first public release of the Vext model family — a line of language models built by [Vext Labs Inc.](https://tryvext.com) specifically for autonomous penetration testing. This 7B release is the foundation: it interprets output from 25+ security tools, plans multi-step attack strategies, classifies vulnerabilities, and generates remediation guidance.

We're actively building our own model from the ground up. v1.1 is the first public checkpoint, trained on proprietary data from real autonomous agent engagements. Future versions are in active training now, scaling toward our flagship model at 100B+ parameters.

What makes this model different: it's trained on real data from autonomous pentesting agents — not internet scrapes or hand-written examples. The [VEXT](https://tryvext.com) platform runs hybrid agents (each with its own browser, CLI tools, and LLM reasoning loop) against authorized targets. Every tool execution, planning decision, and validated finding from those agents feeds back into the training data. The model improves continuously from real-world engagements.

## What This Model Does

| Capability | Description |
|---|---|
| **Tool Output Parsing** | Interprets raw stdout/stderr from 25+ security tools and extracts actionable findings |
| **Attack Planning** | Given recon data, determines which tools to run next, in what order, and with what parameters |
| **Vulnerability Classification** | Distinguishes true positives from false positives across scan results |
| **Remediation Guidance** | Generates actionable fix recommendations for confirmed vulnerabilities |

### Supported Tools

The model understands output from:

`nuclei` · `sqlmap` · `nmap` · `nikto` · `masscan` · `httpx` · `amass` · `gobuster` · `gospider` · `hakrawler` · `ffuf` · `katana` · `subfinder` · `dnsx` · `wpscan` · `sslyze` · `testssl` · `paramspider` · `arjun` · `gau` · `waybackurls` · `wfuzz` · `burpsuite` · `dirsearch` · `hydra` · `whatweb` · `wafw00f` · `commix` · `xsstrike` · `dalfox`

---

## Quickstart

### Serving with vLLM (recommended)

```bash
pip install vllm

vllm serve Vext-Labs-Inc/Vext-labs-7B-v1.1 --trust-remote-code --port 8000
```

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "Vext-Labs-Inc/Vext-labs-7B-v1.1",
    "messages": [
        {
            "role": "system",
            "content": "You are an autonomous security testing agent. Analyze tool output and decide next actions."
        },
        {
            "role": "user",
            "content": "Nuclei scan results:\n[critical] CVE-2021-44228 Log4Shell at /api/login\n\nWhat is this and what should I do next?"
        }
    ],
    "temperature": 0.3,
    "max_tokens": 512
})

print(response.json()["choices"][0]["message"]["content"])
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Vext-Labs-Inc/Vext-labs-7B-v1.1",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Vext-Labs-Inc/Vext-labs-7B-v1.1")

messages = [
    {"role": "system", "content": "You are a security testing agent."},
    {"role": "user", "content": "Gobuster found /admin, /api/v1, /backup. Plan the next steps."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### CLI Script

> **See [`run.py`](run.py) for a ready-to-go inference script with argument parsing.**

```bash
pip install -r requirements.txt

python run.py --prompt "Analyze this nmap scan: ..."
python run.py --prompt-file scan_output.txt
python run.py --interactive
```

---

## Training

| Spec | Value |
|------|-------|
| Parameters | 7B |
| Context length | 32,768 tokens |
| Training samples | 436,922 |
| Training steps | 5,000 |
| Final loss | 0.51 |
| Precision | bfloat16 |
| Inference | vLLM (GPU-accelerated) |

### Where the Training Data Comes From

This model is trained on data generated by autonomous pentesting agents — not scraped from the internet and not hand-written. Each training example comes from a real agent session where a hybrid agent (with its own Chromium browser, 25+ CLI tools, and LLM reasoning loop) ran a real penetration test against an authorized target.

The training data includes:

- **Tool execution traces** — Full input/output from 25+ security tools with parsed results and exit codes
- **Attack planning decisions** — Which tool to run next, why, with what parameters, and what alternatives were considered
- **Vulnerability validation** — True positive vs false positive classification with evidence chains
- **Multi-step attack chains** — Full recon-to-exploitation sequences with reasoning at each step
- **Bug bounty outcomes** — Real-world validation signals including severity ratings and payout data

Data was collected from authorized testing against intentionally vulnerable applications (OWASP Juice Shop, DVWA, bWAPP, WebGoat, and others) and authorized bug bounty targets.

### Continuous Improvement

The model improves continuously through an automated feedback loop. Every agent run captures structured training data — LLM inferences, agent decisions, and finding outcomes — each automatically quality-scored on relevance, accuracy, and efficiency. When enough high-quality examples accumulate, fine-tuning runs automatically on GPU and the updated weights deploy to production. The model that runs the next pentest is better than the one that ran the last.

### Model Roadmap

Vext-labs-7B-v1.1 is the beginning, not the end. We're actively training the next generations of the Vext model:

| Version | Status | Details |
|---------|--------|---------|
| **v1.1 (7B)** | ✅ Released | This release — 32K context, 436K training samples |
| **v1.2** | 🔧 In training | Next iteration with expanded training data from ongoing engagements |
| **Flagship** | 🗺️ Planned | 100B+ parameter model — purpose-built for security from the ground up |

Every pentest our agents run generates new training data. The model compounds — each version is trained on everything the previous versions discovered.

### Additional Training Sources

- **MITRE ATT&CK** — Tactics, techniques, and procedures
- **NVD CVE database** — 240K+ vulnerability records
- **HackerOne disclosed reports** — 10K+ public bug bounty reports
- **OWASP** — Testing guides and cheat sheets
- **Nuclei templates** — Thousands of vulnerability detection templates

---

## Benchmarks

Evaluated across 306 autonomous runs against 26 targets. Each run deploys up to 15 agents with zero human intervention — agents decide which tools to run, interpret results, classify vulnerabilities, and report findings.

### Aggregate Performance

| Metric | Result |
|--------|--------|
| Autonomous runs completed | 306 |
| Targets tested | 26 |
| Total findings generated | 1,977 |
| Validated (true positive) | 139 |
| Unique vulnerability types | 77 |
| OWASP categories covered | 8 / 10 |

### Validated Findings by Severity

| Severity | Count | Examples |
|----------|-------|----------|
| **CRITICAL** | 6 | SQL Injection (CVE-2022-32028) |
| **HIGH** | 23 | Local File Inclusion (CVE-2019-6799), Reflected XSS, DOM XSS |
| **MEDIUM** | 110 | XSS variants, CSRF, GraphQL introspection, info disclosure |
| **Total** | **139** | |

### Per-Target Results

| Target | Type | Runs | Validated | Crit | High | Med |
|--------|------|------|-----------|------|------|-----|
| Acunetix TestPHP | External | 57 | 90 | 6 | 10 | 74 |
| OWASP Juice Shop | CTF | 92 | 26 | — | 13 | 13 |
| Acunetix REST API | External | 12 | 8 | — | — | 8 |
| PortSwigger Gin & Juice | External | 211 | 4 | — | — | 4 |
| tiredful-api | CTF | 8 | 3 | — | — | 3 |
| dvgql (GraphQL) | CTF | 5 | 3 | — | — | 3 |
| Cipher (JWT) | CTF | 11 | 2 | — | — | 2 |
| dvga (GraphQL) | CTF | 4 | 2 | — | — | 2 |
| Zero Bank | External | 77 | 1 | — | — | 1 |

### OWASP Top 10 Coverage

| # | Category | Status |
|---|----------|--------|
| A01 | Broken Access Control | Detected |
| A02 | Cryptographic Failures | Detected |
| A03 | Injection | Detected (SQLi, XSS, Command Injection) |
| A04 | Insecure Design | Detected |
| A05 | Security Misconfiguration | Detected |
| A06 | Vulnerable Components | Detected (CVE matching) |
| A07 | Authentication Failures | Detected |
| A09 | Logging & Monitoring | Detected |

All testing performed against intentionally vulnerable applications and authorized bug bounty targets.

---

## Responsible Use

> **⚠️ This model is for authorized security testing only.**

- Penetration testing with explicit written permission
- CTF competitions and security training environments
- Authorized vulnerability research
- Defensive security analysis

**Do not use this model for unauthorized access to computer systems.**

## Contributing

We welcome contributions — especially around tool parsing support, classification accuracy, evaluation results, and documentation. Open an issue or submit a pull request.

## About Vext Labs

[Vext Labs Inc.](https://tryvext.com) builds autonomous penetration testing agents. Our agents combine LLM reasoning with real browsers and security tools to run full pentests — from reconnaissance to exploitation to reporting — with zero human intervention.

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{vext-labs-7b-v1.1,
  title={Vext-labs-7B-v1.1: A Language Model for Autonomous Penetration Testing},
  author={Vext Labs Inc.},
  year={2026},
  url={https://github.com/Vext-Labs-Inc/Vext-labs-7B-v1.1-}
}
```

---

<p align="center">Built by <a href="https://tryvext.com">Vext Labs Inc.</a></p>
