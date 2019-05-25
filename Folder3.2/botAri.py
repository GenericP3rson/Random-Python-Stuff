# 580946052052418571
# NTgwOTQ2MDUyMDUyNDE4NTcx.XOYG9g.7baRiAYMeVoaTP0xuVnhBpOFKGk
# 67648
# https://discordapp.com/oauth2/authorize?client_id=580946052052418571&scope=bot&permissions=67648

import discord

client = discord.Client()

@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

client.run("https://discordapp.com/oauth2/authorize?client_id=580946052052418571&scope=bot&permissions=67648")
