import discord
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Hugging Face model (DialoGPT)
model_name = "microsoft/DialoGPT-medium"  # Choose "small", "medium", or "large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up Discord bot
intents = discord.Intents.default()
intents.messages = True
bot = discord.Client(intents=intents)

# Keywords to trigger bot responses
CHESS_KEYWORDS = ["chess", "blunder", "checkmate", "pawn", "rook", "bishop", "queen", "king", "endgame", "opening"]

# Generate a response using DialoGPT
def generate_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

# Event listener for new messages
@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if the message contains chess-related keywords
    if any(keyword in message.content.lower() for keyword in CHESS_KEYWORDS):
        user_input = message.content
        response = generate_response(user_input)
        await message.channel.send(response)

    # Optional: respond if the bot is mentioned
    if bot.user.mentioned_in(message):
        user_input = message.content
        response = generate_response(user_input)
        await message.channel.send(response)

# Event to confirm bot is ready
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

# Run the bot (replace YOUR_DISCORD_TOKEN with your token)
bot.run('YOUR_DISCORD_TOKEN')
