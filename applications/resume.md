# Nasiru Abubakar
elnasirulabaran212@gmail.com · Bauchi, Nigeria · github.com/Eienel

---

## Education

**Abubakar Tafawa Balewa University Bauchi**, Nigeria
B.Eng Computer Engineering · 3rd Year · Expected graduation: 2028

---

## Skills

**AI / ML:** Python, PyTorch, HuggingFace Transformers, vLLM, llama.cpp, LLM prompt engineering, few/many-shot ICL, self-consistency, RAG, BAAI/bge-m3 retrieval, structured output parsing
**Full-Stack:** TypeScript, JavaScript, React 18, Next.js 14, Three.js (react-three-fiber), Tailwind CSS, Zustand, Supabase
**Blockchain / Smart Contracts:** Solidity, Foundry, EVM (Base / Arbitrum / Initia), Sui Move, ERC-721, DLMM/AMM protocols, FHE (iExec Nox / Zama)
**Backend / Infra:** Node.js, Vercel (serverless + Cron), Postgres, Drizzle ORM, AES-256-GCM, Next.js App Router
**Tools:** pytest, Git, Bash, Linux, YAML config-as-API

---

## Projects

### FlagOS Open Computing Global Challenge — Track 3 (2025–2026)
`github.com/Eienel/opentract`

Config-driven, format-agnostic In-Context Learning annotation pipeline using Qwen3-4B, built for a global AI leaderboard competition (deadline May 2026). Adapts to any new dataset with a single YAML file edit — no code changes.

- Dense retrieval over labeled exemplar pool (BAAI/bge-m3) to select semantically relevant few-shot examples per input
- Self-consistency: N samples + majority vote for label stability across uncertain inputs
- Long-context chunking and aggregation for over-length inputs
- Structured output parser with bounded re-prompt on malformed outputs
- Automated sweep harness for k-shots × strategy × ordering × thinking-mode — writes winning config automatically
- Multi-backend: hosted API (OpenRouter/DashScope), cloud GPU (vLLM/Transformers), CPU (llama.cpp Q4_K_M)
- Hermetic offline test suite — full E2E verifiable with zero GPU/API/network

**Stack:** Python, PyTorch, HuggingFace Transformers, vLLM, llama.cpp, bge-m3, pytest, YAML

---

### BlockBuilders — Sui Hackathon (2026)
`github.com/Eienel/BlockBuilders`

Educational Web3 game that teaches cryptocurrency concepts through 3D town-building, with AI tutoring and on-chain NFT minting on Sui. Full production-ready application.

- Instanced 3D rendering for thousands of blocks at 60fps (Three.js / react-three-fiber)
- Gemini 2.5 Flash AI tutor agent answers free-form player questions in game context (Vercel Functions)
- Sui Move smart contracts for "Crypto 101" NFT minting on lesson completion; Walrus for off-chain town data
- Public town pages (`/town/<address>`), remix/copy system, leaderboard — full multi-user social layer
- Sandbox free-build mode unlocked after lesson 3 with creative construction tools

**Stack:** TypeScript, React 18, Vite, Three.js, Tailwind, Zustand, Sui blockchain, Gemini 2.5 Flash

---

### ShadowPay — Confidential Payroll Protocol (2026)
`github.com/Eienel/emag`

On-chain payroll protocol using Fully Homomorphic Encryption (FHE) to keep salary amounts private on a public blockchain. Built on Arbitrum with iExec Nox / Zama FHE coprocessor.

- ERC-7984 encrypted stablecoin wrappers — salary amounts hidden from the public ledger, visible only to employer/employee
- ChainGPT natural language interface converts "Pay Alice $5K monthly for 12 months" into on-chain contract parameters
- Chunked vesting streams with cliff support; selective auditor/regulator read-access without full disclosure
- Live demo deployed on Arbitrum Sepolia with employer, employee, and auditor interfaces

**Stack:** Solidity, TypeScript, Next.js 14, wagmi/viem, RainbowKit, iExec Nox, Zama FHE, Arbitrum

---

### Bequest — Crypto Inheritance Dead-Man's Switch (2026)
`github.com/Eienel/getit`

Automated crypto wallet inheritance system — designated assets transfer to beneficiaries if the owner stops checking in. Designed with minimal-trust, scoped-permission architecture.

- Per-bequest hot wallets with AES-256-GCM encrypted private keys; each wallet has restricted agent permissions locked to chain + recipient
- State machine (draft → armed → triggered) enforced by minutely Vercel Cron watchers
- Multi-channel check-in: web button, Telegram webhook, one-tap magic-link email
- Automated nudges at 50/75/90% of check-in window expiry + 30-day policy backstop

**Stack:** Next.js 14, TypeScript, Postgres, Drizzle ORM, Vercel Cron, Zerion API, EVM/Base, Resend, Telegram

---

### Kelvin — DLMM AppChain on Initia (2026)
`github.com/Eienel/Kelvin`

Bin-based Decentralized Liquidity Market Maker deployed as a standalone Initia rollup. Ports Trader Joe's Liquidity Book v2.1.1 to MiniEVM with custom extensions.

- KelvinPositionManager: ERC-721 LP position NFTs with atomic fee collection and rebalancing in a single tx
- One-command rollup deployment and daemon management (shell + tmux)
- Initia session-based transaction signing and `.init` username binding for LP leaderboard

**Stack:** Solidity, Foundry, TypeScript, Next.js, Tailwind, Initia MiniEVM

---

## About

Third-year engineering student at a Nigerian federal university. Self-directed builder — shipped five production-quality projects spanning AI/ML systems, 3D interactive apps, blockchain protocols, and confidential computing. Most recent focus: LLM inference pipelines (RAG, self-consistency, prompt engineering) for a global AI competition. Looking for AI training / RLHF / annotation work where strong software engineering and hands-on LLM knowledge translate directly into task quality.
