setup = """mkdir -p ~/.streamlit/
echo \"[server]
headless = true
port = $PORT
enableCORS = false
\" > ~/.streamlit/config.toml
"""

with open('../setup.sh', 'w', encoding='utf-8') as f:
    f.write(setup)

print("setup.sh created!")