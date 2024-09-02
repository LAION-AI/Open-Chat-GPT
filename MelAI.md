<h1 align="center">
    <span>Open-Assistant</span>
  <img width="auto" height="50px" src="https://github.com/LAION-AI/Open-Assistant/blob/main/assets/logo_crop.png"/>
</h1>

<blockquote>
<p>:memo: <strong>NOTE</strong>: OpenAssistant is completed, and the project is now finished. Thank you to everyone who contributed! Check out our <a href="https://projects.laion.ai/Open-Assistant/blog/2023/10/25/open-assistant-is-completed">blog post</a> for more information. The final published oasst2 dataset can be found on HuggingFace at <a href="https://huggingface.co/datasets/OpenAssistant/oasst2">OpenAssistant/oasst2</a></p>
</blockquote>

<div align="center">

<a href="https://github.com/LAION-AI/Open-Assistant/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/LAION-AI/Open-Assistant?style=social)</a>
<a href="https://laion-ai.github.io/Open-Assistant/">![Docs](https://img.shields.io/badge/docs-laion--ai.github.io%2FOpen--Assistant%2F-green)</a>
<a href="https://github.com/LAION-AI/Open-Assistant/actions/workflows/build-frontend.yaml">![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LAION-AI/Open-Assistant/build-frontend.yaml?label=build-frontend)</a>
<a href="https://github.com/LAION-AI/Open-Assistant/actions/workflows/build-postgres.yaml">![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LAION-AI/Open-Assistant/build-postgres.yaml?label=build-postgres)</a>
<a href="https://github.com/LAION-AI/Open-Assistant/actions/workflows/pre-commit.yaml">![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LAION-AI/Open-Assistant/pre-commit.yaml?label=pre-commit)</a>
<a href="https://github.com/LAION-AI/Open-Assistant/actions/workflows/test-api-contract.yaml">![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LAION-AI/Open-Assistant/test-api-contract.yaml?label=tests-api)</a>
<a href="https://github.com/LAION-AI/Open-Assistant/actions/workflows/test-e2e.yaml">![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LAION-AI/Open-Assistant/test-e2e.yaml?label=tests-web)</a>
<a href="https://github.com/LAION-AI/Open-Assistant/actions/workflows/deploy-docs-site.yaml">![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LAION-AI/Open-Assistant/deploy-docs-site.yaml?label=deploy-docs)</a>
<a href="https://github.com/LAION-AI/Open-Assistant/actions/workflows/production-deploy.yaml">![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LAION-AI/Open-Assistant/production-deploy.yaml?label=deploy-production)</a>
<a href="https://github.com/LAION-AI/Open-Assistant/actions/workflows/release.yaml">![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LAION-AI/Open-Assistant/release.yaml?label=deploy-release)</a>
<a href="https://github.com/LAION-AI/Open-Assistant/releases">![GitHub release (latest by date)](https://img.shields.io/github/v/release/LAION-AI/Open-Assistant)</a>
<a href="https://github-com.translate.goog/LAION-AI/Open-Assistant/blob/main/README.md?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp">![Translate](https://img.shields.io/badge/Translate-blue)</a>

</div>

# Table of Contents

- [What is Open Assistant?](#what-is-open-assistant)
- [Useful Links](#useful-links)
- [How To Try It Out](#how-to-try-it-out)
- [The Vision](#the-vision)
- [The Plan](#the-plan)
- [How You Can Help](#how-you-can-help)

---

## What is Open Assistant?

<p align="center">
Open Assistant is a project meant to give everyone access to a great chat based
large language model.
</p>

We believe that by doing this we will create a revolution in innovation in
language. In the same way that stable-diffusion helped the world make art and
images in new ways we hope Open Assistant can help improve the world by
improving language itself.

# Useful Links

- [Data Collection](https://open-assistant.io)

- [Chat](https://open-assistant.io/chat)

- [Project Documentation](https://projects.laion.ai/Open-Assistant/)

## How To Try It Out

### Chatting with the AI

The chat frontend is now live [here](https://open-assistant.io/chat). Log in and
start chatting! Please try to react with a thumbs up or down for the assistant's
responses when chatting.

### Contributing to Data Collection

The data collection frontend is now live [here](https://open-assistant.io/). Log
in and start taking on tasks! We want to collect a high volume of quality data.
By submitting, ranking, and labelling model prompts and responses you will be
directly helping to improve the capabilities of Open Assistant.

### Running the Development Setup Locally (without chat)

**You do not need to run the project locally unless you are contributing to the
development process. The website link above will take you to the public website
where you can use the data collection app and the chat.**

If you would like to run the data collection app locally for development, you
can set up an entire stack needed to run **Open-Assistant**, including the
website, backend, and associated dependent services, with Docker.

To start the demo, run this in the root directory of the repository (check
[this FAQ](https://projects.laion.ai/Open-Assistant/docs/faq#docker-compose-instead-of-docker-compose)
if you have problems):

```sh
docker compose --profile ci up --build --attach-dependencies
```

> **Note:** when running on MacOS with an M1 chip you have to use:
> `DB_PLATFORM=linux/x86_64 docker compose ...`

Then, navigate to `http://localhost:3000` (It may take some time to boot up) and
interact with the website.

> **Note:** If an issue occurs with the build, please head to the
> [FAQ](https://projects.laion.ai/Open-Assistant/docs/faq) and check out the
> entries about Docker.

> **Note:** When logging in via email, navigate to `http://localhost:1080` to
> get the magic email login link.

> **Note:** If you would like to run this in a standardized development
> environment (a
> ["devcontainer"](https://code.visualstudio.com/docs/devcontainers/containers))
> using
> [vscode locally](https://code.visualstudio.com/docs/devcontainers/create-dev-container#_create-a-devcontainerjson-file)
> or in a web browser using
> [GitHub Codespaces](https://github.com/features/codespaces), you can use the
> provided [`.devcontainer`](.devcontainer/) folder.

### Running the Development Setup Locally for Chat

**You do not need to run the project locally unless you are contributing to the
development process. The website link above will take you to the public website
where you can use the data collection app and the chat.**

**Also note that the local setup is only for development and is not meant to be
used as a local chatbot, unless you know what you are doing.**

If you _do_ know what you are doing, then see the `inference` folder for getting
the inference system up and running, or have a look at `--profile inference` in
addition to `--profile ci` in the above command.

## The Vision

We are not going to stop at replicating ChatGPT. We want to build the assistant
of the future, able to not only write email and cover letters, but do meaningful
work, use APIs, dynamically research information, and much more, with the
ability to be personalized and extended by anyone. And we want to do this in a
way that is open and accessible, which means we must not only build a great
assistant, but also make it small and efficient enough to run on consumer
hardware.

## The Plan

##### We want to get to an initial MVP as fast as possible, by following the 3-steps outlined in the [InstructGPT paper](https://arxiv.org/abs/2203.02155)

1. Collect high-quality human generated Instruction-Fulfillment samples
   (prompt + response), goal >50k. We design a crowdsourced process to collect
   and reviewed prompts. We do not want to train on
   flooding/toxic/spam/junk/personal information data. We will have a
   leaderboard to motivate the community that shows progress and the most active
   users. Swag will be given to the top-contributors.
2. For each of the collected prompts we will sample multiple completions.
   Completions of one prompt will then be shown randomly to users to rank them
   from best to worst. Again this should happen crowd-sourced, e.g. we need to
   deal with unreliable potentially malicious users. At least multiple votes by
   independent users have to be collected to measure the overall agreement. The
   gathered ranking-data will be used to train a reward model.
3. Now follows the RLHF training phase based on the prompts and the reward
   model.

We can then take the resulting model and continue with completion sampling step
2 for a next iteration.

### Slide Decks

[Vision & Roadmap](https://docs.google.com/presentation/d/1n7IrAOVOqwdYgiYrXc8Sj0He8krn5MVZO_iLkCjTtu0/edit?usp=sharing)

[Important Data Structures](https://docs.google.com/presentation/d/1iaX_nxasVWlvPiSNs0cllR9L_1neZq0RJxd6MFEalUY/edit?usp=sharing)

## How You Can Help

All open source projects begin with people like you. Open source is the belief
that if we collaborate we can together gift our knowledge and technology to the
world for the benefit of humanity.

Check out our [contributing guide](CONTRIBUTING.md) to get started.
