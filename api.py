import asyncio
import time
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional
import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InteractionMetrics:
    timestamp: str
    session_id: str
    user_utterance: str
    assistant_response: str
    eou_delay: float    # End of utterance to processing start
    ttft: float         # Time to first token
    ttfb: float         # Time to first byte (TTS start)
    total_latency: float  # Total response time
    stt_duration: float
    llm_duration: float
    tts_duration: float
    tokens_input: int
    tokens_output: int
    processing_successful: bool
    error_message: Optional[str] = None

class MetricsCollector:
    def __init__(self):
        self.interactions: List[InteractionMetrics] = []
        self.session_start_time = time.time()
        self.total_interactions = 0
        self.successful_interactions = 0
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        
    def add_interaction(self, metrics: InteractionMetrics):
        self.interactions.append(metrics)
        self.total_interactions += 1
        if metrics.processing_successful:
            self.successful_interactions += 1
        self.total_tokens_input += metrics.tokens_input
        self.total_tokens_output += metrics.tokens_output
        
        # Log if latency exceeds 2s threshold
        if metrics.total_latency > 2.0:
            logger.warning(f"High latency detected: {metrics.total_latency:.2f}s for interaction: {metrics.user_utterance[:50]}...")
    
    def get_summary_stats(self):
        if not self.interactions:
            return {}
            
        latencies = [i.total_latency for i in self.interactions if i.processing_successful]
        eou_delays = [i.eou_delay for i in self.interactions if i.processing_successful]
        ttfts = [i.ttft for i in self.interactions if i.processing_successful]
        ttfbs = [i.ttfb for i in self.interactions if i.processing_successful]
        
        return {
            'session_duration': time.time() - self.session_start_time,
            'total_interactions': self.total_interactions,
            'successful_interactions': self.successful_interactions,
            'success_rate': self.successful_interactions / self.total_interactions if self.total_interactions > 0 else 0,
            'avg_total_latency': sum(latencies) / len(latencies) if latencies else 0,
            'max_total_latency': max(latencies) if latencies else 0,
            'min_total_latency': min(latencies) if latencies else 0,
            'avg_eou_delay': sum(eou_delays) / len(eou_delays) if eou_delays else 0,
            'avg_ttft': sum(ttfts) / len(ttfts) if ttfts else 0,
            'avg_ttfb': sum(ttfbs) / len(ttfbs) if ttfbs else 0,
            'total_tokens_input': self.total_tokens_input,
            'total_tokens_output': self.total_tokens_output,
            'latency_under_2s_rate': len([l for l in latencies if l < 2.0]) / len(latencies) if latencies else 0
        }
    
    def export_to_excel(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"livekit_metrics_{timestamp}.xlsx"
        
        logger.info(f"Attempting to export metrics to: {filename}")
        logger.info(f"Number of interactions to export: {len(self.interactions)}")

        if not self.interactions:
            logger.warning("No interactions recorded to export to Excel. Skipping file creation.")
            return None

        try:
            # Convert interactions to DataFrame
            df_interactions = pd.DataFrame([asdict(interaction) for interaction in self.interactions])
            logger.info(f"Interactions DataFrame created with {len(df_interactions)} rows.")
            
            # Create summary statistics
            summary_stats = self.get_summary_stats()
            df_summary = pd.DataFrame([summary_stats])
            logger.info("Summary DataFrame created.")
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_interactions.to_excel(writer, sheet_name='Interactions', index=False)
                logger.info("Interactions sheet written.")
                
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                logger.info("Summary sheet written.")
                
                # Add latency analysis sheet
                latency_data = []
                for i, interaction in enumerate(self.interactions):
                    if interaction.processing_successful:
                        latency_data.append({
                            'interaction_id': i + 1,
                            'total_latency': interaction.total_latency,
                            'under_2s_threshold': interaction.total_latency < 2.0,
                            'eou_delay': interaction.eou_delay,
                            'ttft': interaction.ttft,
                            'ttfb': interaction.ttfb,
                            'stt_duration': interaction.stt_duration,
                            'llm_duration': interaction.llm_duration,
                            'tts_duration': interaction.tts_duration
                        })
                
                if latency_data: # Only write if there's data
                    df_latency = pd.DataFrame(latency_data)
                    df_latency.to_excel(writer, sheet_name='Latency_Analysis', index=False)
                    logger.info(f"Latency Analysis sheet written with {len(df_latency)} rows.")
                else:
                    logger.info("No successful interactions for Latency Analysis sheet.")
            
            logger.info(f"Metrics exported successfully to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export metrics to Excel: {e}", exc_info=True) # exc_info=True to print traceback
            return None

class Assistant(Agent):
    def __init__(self, metrics_collector: MetricsCollector) -> None:
        super().__init__(instructions="""You are a helpful voice AI assistant. 
        Keep responses concise and natural for voice interaction. 
        Respond quickly and efficiently to maintain low latency.""")
        self.metrics_collector = metrics_collector
        self.session_id = f"session_{int(time.time())}" # This will be set on the OptimizedAgentSession

class OptimizedAgentSession(AgentSession):
    def __init__(self, metrics_collector: MetricsCollector, **kwargs):
        super().__init__(**kwargs)
        self.metrics_collector = metrics_collector
        self.current_interaction_start = None
        self.current_eou_time = None
        self.current_user_utterance = "" # Not directly used for user_input in generate_reply_with_metrics yet
        
    async def generate_reply_with_metrics(self, user_input: str = None, instructions: str = None):
        interaction_start = time.time()
        self.current_interaction_start = interaction_start
        
        metrics = InteractionMetrics(
            timestamp=datetime.now().isoformat(),
            session_id=getattr(self, 'session_id', 'unknown'), # Use getattr for safety
            user_utterance=user_input or instructions or "",
            assistant_response="",
            eou_delay=0.0,
            ttft=0.0,
            ttfb=0.0,
            total_latency=0.0,
            stt_duration=0.0,
            llm_duration=0.0,
            tts_duration=0.0,
            tokens_input=0,
            tokens_output=0,
            processing_successful=False
        )
        
        try:
            # Track STT timing if user input exists
            stt_start = time.time()
            if user_input:
                # EOU delay is from when user stops speaking to when we start processing
                if self.current_eou_time:
                    metrics.eou_delay = stt_start - self.current_eou_time
                
            # For STT duration (simulated as we don't have direct access)
            # In a real LiveKit agent, STT duration would be provided by the STT plugin
            metrics.stt_duration = time.time() - stt_start # This will be very small without actual STT processing here
            
            # Track LLM processing
            llm_start = time.time()
            
            # Generate response with timeout for latency control
            # The 'instructions' argument to super().generate_reply is used for initial prompts
            # and 'user_input' for user's spoken words.
            # You should pass the relevant one based on context.
            # If both are present, prioritize user_input, or handle as part of a conversation history.
            
            # For the initial greeting, instructions will be used. For subsequent turns, user_input.
            prompt_for_llm = instructions if instructions else user_input
            if not prompt_for_llm: # Handle cases where neither is provided
                logger.warning("No prompt for LLM generation.")
                return ""

            response_task = super().generate_reply(prompt_for_llm)
            
            try:
                # Set timeout slightly under 2s to allow for TTS processing
                response = await asyncio.wait_for(response_task, timeout=1.5)
                
                llm_end = time.time()
                metrics.llm_duration = llm_end - llm_start
                # TTFT is from interaction start to first token from LLM
                metrics.ttft = llm_end - interaction_start 
                
                # Track TTS timing
                tts_start = time.time()
                # TTFB is from interaction start to first byte of audio (TTS starts playing)
                metrics.ttfb = tts_start - interaction_start
                
                # Simulate TTS completion timing
                # In a real implementation, you'd hook into the TTS pipeline to get actual audio duration
                await asyncio.sleep(0.1)  # Simulated TTS processing for a short response
                tts_end = time.time()
                metrics.tts_duration = tts_end - tts_start
                
                # Calculate total latency
                total_end = time.time()
                metrics.total_latency = total_end - interaction_start
                
                # Extract response text (this would need to be adapted based on actual response format)
                # LiveKit agent's generate_reply returns a Response object, often with .text property
                assistant_text_response = response.text if hasattr(response, 'text') else str(response)
                metrics.assistant_response = assistant_text_response[:200] + "..." if len(assistant_text_response) > 200 else assistant_text_response
                
                # Estimate token usage (in real implementation, get from LLM response metadata)
                metrics.tokens_input = len((user_input or instructions or "").split()) # More realistic estimate
                metrics.tokens_output = len(assistant_text_response.split())
                
                metrics.processing_successful = True
                
                # Log performance
                logger.info(f"Response generated - Total latency: {metrics.total_latency:.2f}s, "
                            f"TTFT: {metrics.ttft:.2f}s, TTFB: {metrics.ttfb:.2f}s")
                
                return response
                
            except asyncio.TimeoutError:
                metrics.error_message = "Response generation timeout"
                metrics.total_latency = time.time() - interaction_start
                logger.warning("Response generation timed out")
                
                # Return a fallback response
                fallback_response_text = "I apologize, but I'm experiencing some delay. How can I help you?"
                metrics.assistant_response = fallback_response_text
                
                # Create a dummy response object if the original timed out, to match expected return type
                class TimedOutResponse:
                    def __init__(self, text):
                        self.text = text
                return TimedOutResponse(fallback_response_text)
                
        except Exception as e:
            metrics.error_message = str(e)
            metrics.total_latency = time.time() - interaction_start
            logger.error(f"Error in response generation: {e}")
            # Return an empty response or a generic error object if an unhandled exception occurs
            return "" 
            
        finally:
            self.metrics_collector.add_interaction(metrics)

async def entrypoint(ctx: agents.JobContext):
    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    
    # Optimized configuration for low latency
    session = OptimizedAgentSession(
        metrics_collector=metrics_collector,
        stt=deepgram.STT(
            model="nova-2",   # Using nova-2 for better speed vs nova-3
            language="multi",
            # Add performance optimizations
            interim_results=True,
            smart_format=False,   # Disable for speed
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
            # Optimize for speed
            temperature=0.7,
            # max_tokens=150,  # This should be passed to the generate_reply method if supported
            timeout=1.0,  # Aggressive timeout for LLM
        ),
        tts=cartesia.TTS(
            model="sonic-2", 
            voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
            # Add TTS optimizations if available
            speed=1.1,    # Slightly faster speech
        ),
        vad=silero.VAD.load(
            # Optimize VAD for responsiveness
            # threshold=0.6,   # More sensitive to reduce EOU delay
        ),
        turn_detection=MultilingualModel(),
    )
    
    # Store session reference for metrics
    # Using the agent's session_id as the session_id for metrics
    # Make sure this is set on the OptimizedAgentSession instance itself
    # and not on the Agent instance.
    session.session_id = metrics_collector.session_start_time # Using the timestamp as session_id for uniqueness
    
    await session.start(
        room=ctx.room,
        agent=Assistant(metrics_collector), # Pass the metrics collector to the Assistant too if it needs it directly
        room_input_options=RoomInputOptions(
            # Enable noise cancellation if using LiveKit Cloud for better STT accuracy
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    await ctx.connect()
    
    # Generate initial greeting with metrics tracking
    await session.generate_reply_with_metrics(
        instructions="Greet the user briefly and offer your assistance. Keep it concise for voice interaction."
    )
    
    # IMPORTANT: Keep the session running to process interactions
    # The session's main loop runs here until the room closes or the agent stops
    try:
        await session.run() 
    except Exception as e:
        logger.error(f"Agent session encountered an error during run: {e}")
    finally:
        logger.info("Agent session finished or encountered an unhandled exception during run.")
    
    # Set up cleanup handler
    async def cleanup_handler():
        # Export metrics when session ends
        summary = metrics_collector.get_summary_stats()
        
        logger.info("=== SESSION SUMMARY ===")
        logger.info(f"Session Duration: {summary.get('session_duration', 0):.2f}s")
        logger.info(f"Total Interactions: {summary.get('total_interactions', 0)}")
        logger.info(f"Successful Interactions: {summary.get('successful_interactions', 0)}")
        logger.info(f"Success Rate: {summary.get('success_rate', 0):.2%}")
        logger.info(f"Average Total Latency: {summary.get('avg_total_latency', 0):.2f}s")
        logger.info(f"Latency Under 2s Rate: {summary.get('latency_under_2s_rate', 0):.2%}")
        logger.info(f"Average TTFT: {summary.get('avg_ttft', 0):.2f}s")
        logger.info(f"Average TTFB: {summary.get('avg_ttfb', 0):.2f}s")
        logger.info(f"Total Input Tokens: {summary.get('total_tokens_input', 0)}")
        logger.info(f"Total Output Tokens: {summary.get('total_tokens_output', 0)}")
        
        # Export to Excel
        filename = metrics_collector.export_to_excel()
        if filename:
            logger.info(f"Detailed metrics saved to: {filename}")
        else:
            logger.error("Failed to export metrics to Excel.")
            
    # Register cleanup - ensure this is registered before the session starts running indefinitely
    ctx.add_shutdown_callback(cleanup_handler)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
