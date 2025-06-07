import asyncio
import logging
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.logging import RichHandler
import click
from agent.host import ScientificPaperAgent

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class PaperScoutCLI:
    def __init__(self):
        self.agent = None
        self.running = True
    
    async def initialize(self):
        """Initialize the agent"""
        console.print(Panel.fit("🔬 Scientific Paper Scout", style="bold blue"))
        console.print("Initializing agent...", style="yellow")
        
        try:
            self.agent = ScientificPaperAgent()
            await self.agent.initialize()
            console.print("✅ Agent initialized successfully!", style="green")
            
        except Exception as e:
            console.print(f"❌ Failed to initialize agent: {e}", style="red")
            sys.exit(1)
    
    async def run_chat(self):
        """Main chat loop"""
        console.print("\n" + "="*60)
        console.print("Welcome to Scientific Paper Scout!", style="bold green")
        console.print("I can help you discover and summarize research papers.")
        console.print("Type 'help' for commands or 'quit' to exit.")
        console.print("="*60 + "\n")
        
        while self.running:
            try:
                # Get user input
                user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'log':
                    self.show_tool_log()
                    continue
                elif user_input.lower() == 'clear':
                    console.clear()
                    continue
                
                # Process the message
                console.print("\n[bold green]Assistant:[/bold green]", end=" ")
                
                response_text = ""
                async for chunk in self.agent.process_message(user_input):
                    console.print(chunk, end="", highlight=False)
                    response_text += chunk
                
                console.print()  # New line after response
                
            except KeyboardInterrupt:
                console.print("\n\nInterrupted by user", style="yellow")
                break
            except Exception as e:
                console.print(f"\n❌ Error: {e}", style="red")
                logger.exception("Error in chat loop")
    
    def show_help(self):
        """Show help information"""
        help_text = """
Available Commands:
- help     - Show this help message
- log      - Show tool call log
- clear    - Clear the screen
- quit     - Exit the application

Examples:
- "Search for papers about quantum computing"
- "Find recent research on machine learning"
- "Summarize this paper: https://arxiv.org/pdf/2301.07041.pdf"
        """
        console.print(Panel(help_text.strip(), title="Help", style="cyan"))
    
    def show_tool_log(self):
        """Show the tool call log"""
        if not self.agent:
            console.print("Agent not initialized", style="red")
            return
        
        log_entries = self.agent.get_tool_call_log()
        
        if not log_entries:
            console.print("No tool calls logged yet", style="yellow")
            return
        
        table = Table(title="Tool Call Log")
        table.add_column("Time", style="cyan")
        table.add_column("Server", style="green")
        table.add_column("Tool", style="blue")
        table.add_column("Latency", style="yellow")
        table.add_column("Status", style="magenta")
        
        for entry in log_entries[-10:]:  # Show last 10 entries
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
            latency = f"{entry['latency_seconds']:.2f}s"
            status = "✅ Success" if entry['success'] else "❌ Failed"
            
            table.add_row(
                timestamp,
                entry['server'],
                entry['tool'],
                latency,
                status
            )
        
        console.print(table)
    
    async def shutdown(self):
        """Shutdown gracefully"""
        console.print("\nShutting down...", style="yellow")
        if self.agent:
            await self.agent.shutdown()
        console.print("Goodbye! 👋", style="green")

@click.command()
@click.option('--debug', is_flag=True, help='Enable debug logging')
def main(debug):
    """Scientific Paper Scout - AI assistant for discovering research papers"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cli = PaperScoutCLI()
    
    async def run():
        try:
            await cli.initialize()
            await cli.run_chat()
        finally:
            await cli.shutdown()
    
    # Run the async main function
    asyncio.run(run())

if __name__ == "__main__":
    main()