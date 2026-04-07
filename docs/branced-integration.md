# BRANCED - AI-Powered Flipbook Assistant

## Overview

BRANCED is a new standalone project built as part of the nexus-agents ecosystem. It provides an end-to-end AI assistant powered by Cerebras LLM that guides users step-by-step through any domain or subject, presented in an interactive flipbook format.

## Repository

- **GitHub**: [https://github.com/mahdi1234-hub/branced](https://github.com/mahdi1234-hub/branced)
- **Live Demo**: [https://branced-alpha.vercel.app](https://branced-alpha.vercel.app)

## Architecture

The project is built with:
- **Next.js 16** (App Router, TypeScript)
- **Cerebras LLM** (Llama 4 Scout 17B) for AI guidance
- **Framer Motion + GSAP** for flipbook page-flip animations
- **Nivo Charts + D3.js** for data visualization
- **Tailwind CSS v4** with NOVERA-inspired luxury theme

## User Flow

1. **Landing Page** - Hero with GSAP text reveal animation
2. **Onboarding Flipbook** - 4-step form with conditional logic, variables, and calculations
3. **AI Guidance Chat** - Interactive conversation with context-aware follow-ups
4. **Results Flipbook** - Analysis with charts (bar, pie, line, radar, gauge), key findings, and recommendations

## Key Features

- Multi-step forms with conditional field visibility
- Priority score calculations based on urgency and experience
- Multiple endings based on form inputs
- Inline chart rendering inside the flipbook
- D3.js animated gauge charts
- Risk level indicators
- Full conversation context maintained across the session
