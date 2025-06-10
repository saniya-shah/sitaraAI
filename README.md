# sitaraAI
A minimalist voice assistant made with Livekit. You can call the assistant Sitara.<br>
This project implements a sophisticated, low-latency voice AI assistant designed for natural, conversational interactions. Built on the LiveKit Agent framework, it integrates state-of-the-art speech-to-text (STT), large language model (LLM), and text-to-speech (TTS) technologies. A key feature is its robust performance metrics collection, providing detailed insights into interaction latency, EOU (End of Utterance) delay, and more, exported directly to an Excel file.
<br>
<br>
The main.py file contains the voice assistant code without metrics. <br>
The api.py file contains the voice assistant code with metrics being saved into an excel file.<br>
Kindly use you openai, cartesia, and deepgram api keys.<br> <br>
Features:
<ul>
<li>Real-time Voice Interaction: Engages in fluid, natural conversations.

<li>Optimized for Low Latency: Configured with advanced VAD, STT, LLM, and TTS models to minimize response times.
</ul>
Comprehensive Performance Metrics:
<ul>
<li>EOU Delay: Time from user stopping speech to processing start.

<li>TTFT (Time To First Token): Latency until the LLM generates its first response token.

<li>TTFB (Time To First Byte): Latency until the TTS engine starts generating audio for the response.

<li>Total Latency: Overall response time for each interaction.

<li>Breakdown of STT, LLM, and TTS durations.

<li>Token usage for input and output.

<li>Success/failure tracking for each interaction.

<li>Automatic Excel Export: All interaction metrics are automatically compiled and saved into a multi-sheet Excel file upon session termination.

<li>Configurable Plugins: Easily swap or fine-tune STT, LLM, TTS, and VAD components.
</ul>
Technologies Used<ul>
<li>LiveKit Agents: Framework for building real-time AI agents.

<li>Deepgram: High-performance Speech-to-Text (STT) (using nova-2 model).

<li>OpenAI: Large Language Model (LLM) for conversational AI (using gpt-4o-mini).

<li>Cartesia: Advanced Text-to-Speech (TTS) (using sonic-2 model).

<li>Silero VAD: Voice Activity Detection for accurate turn-taking.

<li>Pandas: For data manipulation and Excel export.

<li>python-dotenv: For managing environment variables.
</ul>
